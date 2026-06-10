# WaveHands

Gesture-controlled music performance system built with Python, MediaPipe, OpenCV, and low-latency audio synthesis.

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-009688?style=flat-square&logo=google&logoColor=white" alt="MediaPipe">
  <img src="https://img.shields.io/badge/OpenCV-Real--time%20Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Audio-Low%20Latency-f5a623?style=flat-square" alt="Low latency audio">
  <img src="https://img.shields.io/badge/C%20DSP-Optional%20Acceleration-4dccbd?style=flat-square" alt="C DSP">
</p>

## Overview

WaveHands turns hand movement into musical performance. It tracks the user's hand with a webcam, maps gestures to notes on a radial music interface, and synthesizes audio in real time. The project also includes sustain control, a gesture-driven loop station, selectable scales, and multiple instrument modes.

This is the kind of project that combines creative interaction design with real-time engineering constraints.

## Why This Project Matters

- It explores human-computer interaction through gesture-based control.
- It mixes computer vision, audio synthesis, and interface rendering in one real-time application.
- It shows how to structure a performance-oriented Python app with clear modular boundaries.

## Core Features

- Hand tracking with MediaPipe and OpenCV
- Dynamic note wheel for scales such as major, minor, pentatonic, blues, and chromatic
- Real-time monophonic synthesis with selectable instrument behavior
- Sustain pedal behavior
- Gesture-based loop station with recording, playback, and overdubbing
- Responsive UI with camera mode and neutral-background mode
- In-app video recording with optional FFmpeg muxing
- Optional C helper for lower-latency voice mixing

## Musical Interaction Model

WaveHands currently supports:

- Chromatic tonics with sharp or flat note naming
- Major, natural minor, major pentatonic, minor pentatonic, blues, and chromatic scales
- Instrument modes: `Sine`, `Piano`, and `Drums`
- Note triggering through stable hover selection
- Harmonic color changes through a secondary chord/subnote wheel

## Runtime Architecture

The app is organized into a layered structure:

```text
app.py
src/wavehands/
  application/
    app_controller.py
    loop_station.py
    selection_service.py
  domain/
    looping.py
    models.py
    notes.py
  infrastructure/
    camera.py
    hand_tracker.py
    audio/mono_synth.py
    audio/c_voice_mixer.py
  presentation/
    controls.py
    gesture_pedals.py
    note_wheel.py
    renderer.py
  config.py
```

### Main pipeline

1. The camera thread reads frames from the webcam.
2. The tracking layer detects hands and extracts pointer positions.
3. The application layer resolves stable hover selection and loop-state changes.
4. The audio engine maps the selection to frequency data and triggers sound output.
5. The presentation layer renders the wheel, HUD, controls, and performance state.

## Performance Notes

- Default processing resolution is optimized for responsiveness.
- Audio runs through `sounddevice` with a low buffer size.
- The loop voice mixer can use a compiled C helper for better performance.
- The app reports runtime metrics such as FPS, hand-detection ratio, current note, audio callback rate, XRuns, and active voices.

## Requirements

- Python `3.11.10`
- Webcam
- System audio output
- `ffmpeg` if you want exported MP4 files with muxed audio

## Installation

```bash
pyenv local 3.11.10
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional: compile the C helper

```bash
./scripts/build_c_helpers.sh
```

## Run the App

```bash
source .venv/bin/activate
python app.py
```

Useful controls:

- `f` toggles fullscreen mode
- `q` or `ESC` exits the app

## Technical Documentation

- [Technical architecture](docs/technical-architecture.md)
- [Music model, instruments, and feature roadmap](docs/music-instruments-and-features.md)

## What This Project Demonstrates

- Real-time application design under latency constraints
- Gesture-driven interface design
- Modular Python architecture across domain, application, infrastructure, and presentation layers
- Practical integration of computer vision and audio synthesis
- Performance-minded engineering with optional native acceleration

## Current Limitations

- The current MVP is monophonic by design
- Rhythmic quantization is not implemented yet
- Piano and drum synthesis can still be improved for realism
- Some real-time orchestration logic remains concentrated in `app_controller.py`

## Next Improvements

- MIDI export
- Visual and audio metronome
- Quantized loop recording
- Additional instruments and smarter chord behavior
- More test coverage around real-time orchestration