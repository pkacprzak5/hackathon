"""Entry point for python -m squat_coach."""
import argparse
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Squat Coach")
    parser.add_argument("--mode", choices=["webcam", "replay", "train"], default="webcam")
    parser.add_argument("--video", type=str, help="Video file path for replay mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-features", action="store_true", help="Enable per-frame JSONL feature logging")
    args = parser.parse_args()

    from squat_coach.app import SquatCoachApp
    app = SquatCoachApp(
        mode=args.mode,
        video_path=args.video,
        debug=args.debug,
        log_features=args.log_features,
    )
    app.run()

if __name__ == "__main__":
    main()
