# Squat Coach

Real-time squat analysis system using MediaPipe BlazePose 3D landmarks, TCN+GRU temporal models, and structured scoring.

## Setup

```bash
cd squat_coach
pip install -r requirements.txt
```

## Train Models

```bash
python -m squat_coach --mode train
```

This generates synthetic training data, trains TCN and GRU models, and saves checkpoints.

## Run Live Analysis

```bash
python -m squat_coach --mode webcam
```

Stand still for 2 seconds for calibration, then start squatting.

## Replay a Video

```bash
python -m squat_coach --mode replay --video path/to/squat_video.mp4
```

## Debug Mode

```bash
python -m squat_coach --mode webcam --debug
```

## Controls

- Press `q` to quit

## Architecture

Video -> MediaPipe Pose -> Feature Extraction (D=42) -> TCN+GRU Ensemble -> Phase Detection + Fault Detection -> Scoring -> Terminal Logging + Overlay + Gemini Payloads

See `docs/superpowers/specs/2026-03-28-squat-coach-design.md` for the full design specification.
