#!/bin/bash
# Install PortAudio system dependency for PyAudio on Render (Linux)
apt-get update && apt-get install -y portaudio19-dev
pip install -r requirements.txt
