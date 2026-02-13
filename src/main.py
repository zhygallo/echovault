from __future__ import annotations

import argparse
import sys

from src.config import Config
from src.utils.logger import setup_logger
from src.audio.recorder import AudioRecorder
from src.audio.player import AudioPlayer
from src.stt.whisper_stt import WhisperSTT
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


def run_push_to_talk(config: Config):
    """Push-to-talk mode: press Enter to start/stop recording."""
    recorder, player, stt, llm, tts = build_components(config)

    logger.info("Push-to-talk mode. Press Enter to record, Enter to stop.")
    logger.info("Type 'quit' to exit, 'reset' to clear conversation.\n")

    try:
        while True:
            user_input = input("\n> Press Enter to record (or type 'quit'/'reset'): ").strip()

            if user_input.lower() == "quit":
                logger.info("Goodbye!")
                break
            if user_input.lower() == "reset":
                llm.reset_conversation()
                print("Conversation reset.")
                continue

            # Record audio
            audio = recorder.record_until_enter()
            if len(audio) == 0:
                print("No audio recorded.")
                continue

            # Transcribe
            print("Transcribing...")
            text = stt.transcribe(audio, config.audio.sample_rate)
            if not text:
                print("Could not transcribe audio.")
                continue
            print(f"You said: {text}")

            # Get LLM response
            print("Thinking...")
            response = llm.respond(text)
            print(f"Jarvis: {response}")

            # Speak response
            print("Speaking...")
            chunks = tts.synthesize_stream(response)
            player.play_stream(chunks, tts.get_sample_rate())

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        recorder.close()


def run_always_listening(config: Config):
    """Always-listening mode: wake word triggers recording."""
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

    def on_wake_word():
        logger.info("Wake word detected! Listening for command...")

        # Record with VAD
        audio = recorder.record_with_vad(
            aggressiveness=config.vad.aggressiveness,
            silence_timeout=config.vad.silence_timeout,
            frame_duration_ms=config.vad.frame_duration_ms,
        )
        if len(audio) == 0:
            logger.info("No speech detected.")
            return

        # Transcribe
        text = stt.transcribe(audio, config.audio.sample_rate)
        if not text:
            logger.info("Could not transcribe audio.")
            return
        print(f"You said: {text}")

        # Get LLM response
        response = llm.respond(text)
        print(f"Jarvis: {response}")

        # Speak response
        chunks = tts.synthesize_stream(response)
        player.play_stream(chunks, tts.get_sample_rate())

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
    args = parser.parse_args()

    try:
        config = Config.load(args.config)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Starting EchoVault in {args.mode} mode...")

    if args.mode == "push-to-talk":
        run_push_to_talk(config)
    else:
        run_always_listening(config)


if __name__ == "__main__":
    main()
