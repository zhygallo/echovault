from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, render_template
from flask_socketio import SocketIO

from src.ui.event_bus import EventBus
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_template_dir = str(Path(__file__).parent / "templates")
app = Flask(__name__, template_folder=_template_dir)
app.config["SECRET_KEY"] = "echovault"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


@app.route("/")
def index():
    return render_template("index.html")


def start_web_ui(event_bus: EventBus, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Start the web UI in a background daemon thread, wired to the event bus."""

    def _forward(event_name: str):
        def _handler(data):
            socketio.emit(event_name, data)
        return _handler

    for evt in ("status_changed", "user_message", "assistant_message", "conversation_reset"):
        event_bus.on(evt, _forward(evt))

    def _run():
        logger.info(f"Web UI starting on http://{host}:{port}")
        socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True, log_output=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
