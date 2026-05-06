"""
Sign Language Bridge — Two-Way Multilingual Sign Language Communication System
===============================================================================
A real-time, bidirectional communication bridge between sign language users
and hearing individuals. Uses MediaPipe + ML for gesture recognition and
supports multilingual text/speech output.

Usage:
    python main.py                        # Default webcam, English output
    python main.py --lang hi              # Hindi output
    python main.py --lang ta --cam 1      # Tamil output, second camera
    python main.py --no-speech            # Disable text-to-speech
    python main.py --mode text-to-sign    # Text-to-sign only mode
"""

import argparse
import cv2
import time

from utils.hand_detection import HandDetector
from utils.gesture_recognizer import GestureRecognizer
from utils.speech_converter import SpeechConverter
from utils.translator import Translator
from utils.sign_display import SignDisplay
from utils.ui import SignLanguageUI


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sign Language Bridge: Real-Time Bidirectional Communication"
    )
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Output language code, e.g. en, hi, ta, te, fr (default: en)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "sign-to-text", "text-to-sign"],
        default="both",
        help="Operating mode (default: both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/gesture_model.pkl",
        help="Path to trained gesture classifier (default: models/gesture_model.pkl)",
    )
    parser.add_argument(
        "--no-speech", action="store_true", help="Disable text-to-speech output"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.75,
        help="Minimum gesture confidence threshold (default: 0.75)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Initializing Sign Language Bridge...")

    # Initialize modules
    detector = HandDetector()
    recognizer = GestureRecognizer(model_path=args.model, confidence=args.confidence)
    speech = SpeechConverter(lang=args.lang, enabled=not args.no_speech)
    translator = Translator(target_lang=args.lang)
    sign_display = SignDisplay()
    ui = SignLanguageUI(lang=args.lang)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.cam}")
        return

    # State
    last_gesture = None
    sentence_buffer = []
    last_spoken_time = 0
    SPEAK_INTERVAL = 2.5  # seconds between TTS announcements

    print(f"[INFO] Running in '{args.mode}' mode | Language: {args.lang}")
    print("[INFO] Press 'q' to quit | 's' to speak buffer | 'c' to clear buffer")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        landmarks, annotated_frame = detector.detect(frame)

        gesture_label = None
        confidence = 0.0

        # --- Sign-to-Text pipeline ---
        if args.mode in ("both", "sign-to-text") and landmarks:
            gesture_label, confidence = recognizer.predict(landmarks)

            if gesture_label and gesture_label != last_gesture:
                last_gesture = gesture_label

                # Translate if not English
                display_text = (
                    translator.translate(gesture_label)
                    if args.lang != "en"
                    else gesture_label
                )
                sentence_buffer.append(display_text)

                # Auto-speak if interval elapsed
                now = time.time()
                if now - last_spoken_time >= SPEAK_INTERVAL:
                    speech.speak(display_text)
                    last_spoken_time = now

        # --- Text-to-Sign display ---
        text_input = ui.get_text_input()
        sign_frame = None
        if args.mode in ("both", "text-to-sign") and text_input:
            sign_frame = sign_display.get_sign_frame(text_input)

        # --- Render UI ---
        annotated_frame = ui.render(
            frame=annotated_frame,
            gesture=gesture_label,
            confidence=confidence,
            buffer=sentence_buffer,
            sign_frame=sign_frame,
        )

        cv2.imshow("Sign Language Bridge", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and sentence_buffer:
            full_sentence = " ".join(sentence_buffer)
            speech.speak(full_sentence)
            last_spoken_time = time.time()
        elif key == ord("c"):
            sentence_buffer.clear()
            last_gesture = None
            print("[INFO] Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Session ended.")


if __name__ == "__main__":
    main()
