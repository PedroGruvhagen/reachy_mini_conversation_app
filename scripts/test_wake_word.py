#!/usr/bin/env python3
"""Test script for wake word detection.

Live microphone testing of OpenWakeWord integration.
Useful for validating detection and tuning confidence thresholds.

Usage:
    python scripts/test_wake_word.py
    python scripts/test_wake_word.py --model hey_mycroft --threshold 0.6
"""

import sys
import time
import signal
import logging
import argparse
from typing import Optional

import numpy as np

# Try to import sounddevice for microphone access
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

# Add src to path for imports
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reachy_mini_conversation_app.headless.wake_word import (
    WakeWordEngine,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    WAKE_WORD_SAMPLE_RATE,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WakeWordTester:
    """Interactive wake word testing utility."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.5,
        cooldown: float = 2.0,
    ) -> None:
        """Initialize the tester.

        Args:
            model_name: Wake word model to test.
            threshold: Confidence threshold.
            cooldown: Cooldown between detections (shorter for testing).
        """
        self._model_name = model_name
        self._threshold = threshold
        self._cooldown = cooldown
        self._running = False
        self._detection_count = 0
        self._engine: Optional[WakeWordEngine] = None

    def _on_detection(self, confidence: float) -> None:
        """Callback when wake word detected."""
        self._detection_count += 1
        print(f"\n{'='*60}")
        print(f"  WAKE WORD DETECTED! #{self._detection_count}")
        print(f"  Confidence: {confidence:.3f} (threshold: {self._threshold})")
        print(f"  Model: {self._model_name}")
        print(f"{'='*60}\n")

    def run(self) -> None:
        """Run the interactive test."""
        print("\n" + "=" * 60)
        print("  Wake Word Detection Test")
        print("=" * 60)
        print(f"  Model: {self._model_name}")
        print(f"  Threshold: {self._threshold}")
        print(f"  Cooldown: {self._cooldown}s")
        print(f"  Sample Rate: {WAKE_WORD_SAMPLE_RATE}Hz")
        print("=" * 60 + "\n")

        # Initialize engine
        print("Initializing wake word engine...")
        self._engine = WakeWordEngine(
            model_name=self._model_name,
            confidence_threshold=self._threshold,
            cooldown_seconds=self._cooldown,
            on_wake_word=self._on_detection,
        )

        if not self._engine.initialize():
            print("ERROR: Failed to initialize wake word engine!")
            return

        print(f"Engine initialized: {self._engine.get_stats()}")
        print("\nListening for wake word... (Ctrl+C to stop)\n")

        self._running = True

        # Setup signal handler
        def handle_signal(sig: int, frame: object) -> None:
            print("\nStopping...")
            self._running = False

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Audio callback
        frame_count = 0
        last_level_time = time.time()

        def audio_callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            nonlocal frame_count, last_level_time

            if status:
                logger.warning(f"Audio status: {status}")

            # Convert to int16
            audio_int16 = (indata[:, 0] * 32767).astype(np.int16)

            # Process for wake word
            self._engine.process_audio(audio_int16)

            # Show audio level periodically
            frame_count += 1
            if time.time() - last_level_time > 1.0:
                level = np.abs(audio_int16).mean()
                bars = int(level / 1000)
                bar_str = "|" * min(bars, 50)
                print(f"\rAudio level: {bar_str:<50} ({level:>5.0f})  ", end="", flush=True)
                last_level_time = time.time()

        # Start audio stream
        try:
            with sd.InputStream(
                samplerate=WAKE_WORD_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=audio_callback,
            ):
                while self._running:
                    time.sleep(0.1)

        except Exception as e:
            print(f"ERROR: Audio stream failed: {e}")
            return

        # Cleanup
        print("\n")
        self._engine.shutdown()
        print(f"\nTest complete. Total detections: {self._detection_count}")
        print(f"Engine stats: {self._engine.get_stats()}")


def list_models() -> None:
    """Print available models."""
    print("\nAvailable wake word models:")
    print("-" * 40)
    for model in AVAILABLE_MODELS:
        default_marker = " (default)" if model == DEFAULT_MODEL else ""
        print(f"  - {model}{default_marker}")
    print()


def list_audio_devices() -> None:
    """Print available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    print(sd.query_devices())
    print()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test wake word detection with live microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Test with default model (hey_jarvis)
  %(prog)s --model alexa            # Test with alexa model
  %(prog)s --threshold 0.7          # Use higher confidence threshold
  %(prog)s --list-models            # Show available models
  %(prog)s --list-devices           # Show audio devices
        """,
    )

    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Wake word model to test (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.5,
        help="Confidence threshold 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--cooldown", "-c",
        type=float,
        default=2.0,
        help="Cooldown between detections in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available wake word models and exit",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.list_devices:
        list_audio_devices()
        return

    # Validate model
    if args.model not in AVAILABLE_MODELS:
        print(f"ERROR: Unknown model '{args.model}'")
        list_models()
        sys.exit(1)

    # Run test
    tester = WakeWordTester(
        model_name=args.model,
        threshold=args.threshold,
        cooldown=args.cooldown,
    )
    tester.run()


if __name__ == "__main__":
    main()
