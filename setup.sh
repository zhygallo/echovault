#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"

echo "=== EchoVault Setup ==="
echo ""

# Detect OS and install system dependencies
OS="$(uname -s)"
case "$OS" in
    Darwin)
        echo "[1/5] Installing system dependencies via Homebrew..."
        if ! command -v brew &>/dev/null; then
            echo "Error: Homebrew is required on macOS. Install from https://brew.sh"
            exit 1
        fi
        brew install portaudio ffmpeg 2>/dev/null || true
        ;;
    Linux)
        echo "[1/5] Installing system dependencies via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq portaudio19-dev ffmpeg python3-dev
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# Install Poetry if missing
echo "[2/5] Checking for Poetry..."
if ! command -v poetry &>/dev/null; then
    echo "  Poetry not found. Installing via pipx..."
    if ! command -v pipx &>/dev/null; then
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
    fi
    pipx install poetry
fi
echo "  Poetry $(poetry --version)"

# Install Python dependencies via Poetry
echo "[3/5] Installing Python dependencies..."
poetry install

# Download Piper voice model
echo "[4/5] Downloading Piper voice model (en_US-lessac-medium)..."
mkdir -p models
if [ ! -f "models/en_US-lessac-medium.onnx" ]; then
    curl -L -o "models/en_US-lessac-medium.onnx" "$PIPER_VOICE_URL"
    echo "  Downloaded voice model."
else
    echo "  Voice model already exists, skipping."
fi
if [ ! -f "models/en_US-lessac-medium.onnx.json" ]; then
    curl -L -o "models/en_US-lessac-medium.onnx.json" "$PIPER_CONFIG_URL"
    echo "  Downloaded voice config."
else
    echo "  Voice config already exists, skipping."
fi

# Copy example configs if missing
echo "[5/5] Setting up configuration files..."
if [ ! -f "config.yaml" ]; then
    cp config.example.yaml config.yaml
    echo "  Created config.yaml from example."
else
    echo "  config.yaml already exists, skipping."
fi
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  Created .env from example."
    echo ""
    echo "  *** IMPORTANT: Edit .env and add your Anthropic API key ***"
else
    echo "  .env already exists, skipping."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your Anthropic API key"
echo "  2. Run: bash run.sh                    (push-to-talk mode)"
echo "  3. Run: bash run.sh --mode always-listening  (wake word mode)"
