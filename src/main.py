from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.ui.event_bus import EventBus
from src.utils.logger import setup_logger
from src.audio.recorder import AudioRecorder
from src.audio.player import AudioPlayer
from src.llm.claude_llm import ClaudeLLM
from src.tts.piper_tts import PiperTTS

logger = setup_logger("echovault")


def build_components(config: Config):
    recorder = AudioRecorder(
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
        chunk_size=config.audio.chunk_size,
    )
    player = AudioPlayer()

    if config.whisper.backend == "api":
        from src.stt.whisper_api_stt import WhisperAPISTT
        stt = WhisperAPISTT(api_key=config.whisper.api_key)
    else:
        from src.stt.whisper_stt import WhisperSTT
        stt = WhisperSTT(
            model_name=config.whisper.model,
            device=config.whisper.device,
        )

    llm = ClaudeLLM(
        api_key=config.claude.api_key,
        model=config.claude.model,
        max_tokens=config.claude.max_tokens,
        system_prompt=config.claude.system_prompt,
        max_history_pairs=config.claude.max_history_pairs,
    )
    tts = PiperTTS(
        model_path=config.piper.model_path,
        config_path=config.piper.config_path,
    )
    return recorder, player, stt, llm, tts


def run_push_to_talk(config: Config, event_bus: EventBus):
    """Push-to-talk mode: press Enter to start/stop recording."""
    recorder, player, stt, llm, tts = build_components(config)

    logger.info("Push-to-talk mode. Press Enter to record, Enter to stop.")
    logger.info("Type 'quit' to exit, 'reset' to clear conversation.\n")
    event_bus.emit("status_changed", {"status": "idle"})

    try:
        while True:
            user_input = input("\n> Press Enter to record (or type 'quit'/'reset'): ").strip()

            if user_input.lower() == "quit":
                logger.info("Goodbye!")
                break
            if user_input.lower() == "reset":
                llm.reset_conversation()
                event_bus.emit("conversation_reset", {})
                print("Conversation reset.")
                continue

            # Record audio
            event_bus.emit("status_changed", {"status": "recording"})
            audio = recorder.record_until_enter()
            if len(audio) == 0:
                print("No audio recorded.")
                event_bus.emit("status_changed", {"status": "idle"})
                continue

            # Transcribe
            event_bus.emit("status_changed", {"status": "transcribing"})
            print("Transcribing...")
            text = stt.transcribe(audio, config.audio.sample_rate)
            if not text:
                print("Could not transcribe audio.")
                event_bus.emit("status_changed", {"status": "idle"})
                continue
            print(f"You said: {text}")
            event_bus.emit("user_message", {"text": text})

            # Get LLM response
            event_bus.emit("status_changed", {"status": "thinking"})
            print("Thinking...")
            response = llm.respond(text)
            print(f"Jarvis: {response}")
            event_bus.emit("assistant_message", {"text": response})

            # Speak response
            event_bus.emit("status_changed", {"status": "speaking"})
            print("Speaking...")
            chunks = tts.synthesize_stream(response)
            player.play_stream(chunks, tts.get_sample_rate())
            event_bus.emit("status_changed", {"status": "idle"})

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        recorder.close()


def run_always_listening(config: Config, event_bus: EventBus):
    """Always-listening mode: wake word triggers a multi-turn conversation."""
    from src.wakeword.oww_wakeword import OpenWakeWordDetector

    recorder, player, stt, llm, tts = build_components(config)

    detector = OpenWakeWordDetector(
        model_name=config.wakeword.model_name,
        threshold=config.wakeword.threshold,
        chunk_size=config.wakeword.chunk_size,
        sample_rate=config.audio.sample_rate,
    )

    logger.info(f"Always-listening mode. Say '{config.wakeword.model_name}' to activate.")
    logger.info("Press Ctrl+C to exit.\n")
    event_bus.emit("status_changed", {"status": "idle"})

    listen_window = 5.0  # seconds to wait for speech each iteration
    idle_limit = 30.0    # total silence before returning to wake word mode

    def on_wake_word():
        logger.info("Wake word detected! Entering conversation mode...")
        event_bus.emit("status_changed", {"status": "wake"})
        cumulative_silence = 0.0

        while True:
            # Record with VAD, using a short listen window
            event_bus.emit("status_changed", {"status": "listening"})
            audio = recorder.record_with_vad(
                aggressiveness=config.vad.aggressiveness,
                silence_timeout=config.vad.silence_timeout,
                frame_duration_ms=config.vad.frame_duration_ms,
                listen_timeout=listen_window,
            )

            if len(audio) == 0:
                cumulative_silence += listen_window
                logger.info(
                    f"No speech detected ({cumulative_silence:.0f}s/{idle_limit:.0f}s idle)."
                )
                if cumulative_silence >= idle_limit:
                    logger.info("No speech for 30s, returning to wake word mode.")
                    break
                continue

            # Got speech — reset idle counter
            cumulative_silence = 0.0

            # Transcribe
            event_bus.emit("status_changed", {"status": "transcribing"})
            text = stt.transcribe(audio, config.audio.sample_rate)
            if not text:
                logger.info("Could not transcribe audio.")
                continue
            print(f"You said: {text}")
            event_bus.emit("user_message", {"text": text})

            # Get LLM response
            event_bus.emit("status_changed", {"status": "thinking"})
            response = llm.respond(text)
            print(f"Jarvis: {response}")
            event_bus.emit("assistant_message", {"text": response})

            # Speak response
            event_bus.emit("status_changed", {"status": "speaking"})
            chunks = tts.synthesize_stream(response)
            player.play_stream(chunks, tts.get_sample_rate())

        # Exiting conversation — reset history for next activation
        llm.reset_conversation()
        event_bus.emit("conversation_reset", {})
        event_bus.emit("status_changed", {"status": "idle"})
        logger.info("Conversation ended. Listening for wake word...")

    try:
        detector.listen(on_wake_word)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        detector.close()
        recorder.close()


def main():
    parser = argparse.ArgumentParser(description="EchoVault Voice Assistant")
    parser.add_argument(
        "--mode",
        choices=["push-to-talk", "always-listening"],
        default="push-to-talk",
        help="Operating mode (default: push-to-talk)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        default=False,
        help="Enable web UI (default: disabled)",
    )
    args = parser.parse_args()

    try:
        config = Config.load(args.config)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    event_bus = EventBus()

    if args.ui:
        from src.ui.web_ui import start_web_ui
        start_web_ui(event_bus)

    logger.info(f"Starting EchoVault in {args.mode} mode...")

    if args.mode == "push-to-talk":
        run_push_to_talk(config, event_bus)
    else:
        run_always_listening(config, event_bus)


if __name__ == "__main__":
    main()
