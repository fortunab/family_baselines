#!/usr/bin/env bash
# Setup script using Pixi for Linux / WSL / macOS
set -e

if ! command -v pixi &> /dev/null; then
    if [ -f "$HOME/.pixi/bin/pixi" ]; then
        export PATH="$HOME/.pixi/bin:$PATH"
    else
        echo "Pixi not found. Downloading and installing official Pixi binary..."
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
fi

echo "Pixi version: $(pixi --version)"
echo "Installing environment and dependencies with Pixi..."
pixi install
echo "[SUCCESS] Pixi environment ready!"
echo "To run experiment: pixi run train"
echo "To run linting:    pixi run lint"
echo "To enter shell:    pixi shell"
