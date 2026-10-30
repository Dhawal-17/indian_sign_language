import os
import cv2
import numpy as np
from pathlib import Path
from collections import deque, Counter

ROOT = Path(__file__).resolve().parent.parent
if "KERAS_HOME" not in os.environ and (ROOT / ".keras").exists():
    os.environ["KERAS_HOME"] = str((ROOT / ".keras").resolve())
if "MPLCONFIGDIR" not in os.environ and (ROOT / ".keras" / "matplotlib").exists():
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".keras" / "matplotlib").resolve())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from models import load_isl_model, CLASSES, IMG_SIZE

MODEL_PATH = ROOT / "models" / "isl_model.keras"

# Recognition and Smoothing Configuration
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
MARGIN_THRESHOLD = 0.15  # Minimum difference between top-1 and top-2
CONFIRMATION_FRAMES = 24  # ~1 second at 24-30 FPS for hold-to-confirm
EMA_ALPHA = 0.35  # Exponential moving average weight for current frame (0.0 to 1.0)
MIN_HAND_AREA = 3500  # Minimum skin pixels in ROI to consider hand present


class HandDetector:
    """
    Real-time hand detector using robust YCrCb skin segmentation and contour tracking.
    Guarantees no false triggers on empty backgrounds or white space.
    """
    def __init__(self, min_area=MIN_HAND_AREA):
        self.min_area = min_area
        self.lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        self.upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect_in_roi(self, roi_bgr):
        """Detects hand in a region of interest, returning presence, area, and bounding box."""
        if roi_bgr is None or roi_bgr.size == 0:
            return False, 0, None, None

        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_skin, self.upper_skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0, None, mask

        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area < self.min_area:
            return False, area, None, mask

        x, y, w, h = cv2.boundingRect(max_contour)
        return True, area, (x, y, w, h), mask


def draw_styled_rect(img, pt1, pt2, color, thickness=2, corner_len=20):
    """Draws a clean bounding box with accented corner brackets."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)


def main():
    print(f"Loading trained ISL model from: {MODEL_PATH}")
    model = load_isl_model(MODEL_PATH)
    print("Model loaded successfully. Initializing webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not available. Check camera index or permissions.")

    # Try setting optimal capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = HandDetector()

    confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
    history = deque(maxlen=8)
    smoothed_probs = None

    candidate = None
    candidate_frames = 0
    last_confirmed = None
    text_buffer = ""
    flash_frames = 0

    print("\n--- Indian Sign Language Recognition Active ---")
    print("Controls:")
    print("  [Q] or [ESC] : Quit application")
    print("  [C]          : Clear text buffer")
    print("  [B]          : Backspace last letter")
    print("  [Space]      : Add space between words")
    print("  [+] / [-]    : Adjust confidence threshold")
    print("------------------------------------------------\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read webcam frame.")
                break

            # Mirror preview for natural human interaction
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Define Guided Detection ROI (centered vertically on right half)
            roi_size = min(int(h * 0.72), int(w * 0.52))
            roi_x1 = w - roi_size - 30
            roi_y1 = (h - roi_size) // 2
            roi_x2 = roi_x1 + roi_size
            roi_y2 = roi_y1 + roi_size

            roi_bgr = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            hand_present, hand_area, bbox, _ = detector.detect_in_roi(roi_bgr)

            top3_indices = []
            top3_text = ""
            status_text = ""
            status_color = (0, 165, 255)  # Orange: No Hand
            display_letter = "No Hand Detected"
            current_conf = 0.0
            model_input_preview = None

            if hand_present and bbox is not None:
                status_color = (0, 255, 0)  # Green: Hand Active
                status_text = "Hand Tracked"

                # Sub-crop the hand with padding while preserving square aspect ratio
                bx, by, bw, bh = bbox
                pad = int(max(bw, bh) * 0.25)
                cx, cy = bx + bw // 2, by + bh // 2
                half_side = max(bw, bh) // 2 + pad

                sub_x1 = max(0, cx - half_side)
                sub_y1 = max(0, cy - half_side)
                sub_x2 = min(roi_bgr.shape[1], cx + half_side)
                sub_y2 = min(roi_bgr.shape[0], cy + half_side)

                cropped_hand = roi_bgr[sub_y1:sub_y2, sub_x1:sub_x2]
                if cropped_hand.size > 0:
                    # Draw tight bounding box inside ROI
                    cv2.rectangle(
                        frame,
                        (roi_x1 + sub_x1, roi_y1 + sub_y1),
                        (roi_x1 + sub_x2, roi_y1 + sub_y2),
                        (0, 255, 0), 2
                    )

                    # Prepare image for model inference matching data.py
                    rgb_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB)
                    model_input_preview = cv2.resize(cropped_hand, (110, 110))
                    resized_hand = cv2.resize(rgb_hand, (IMG_SIZE, IMG_SIZE))
                    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
                        resized_hand.astype(np.float32)
                    )

                    # Forward pass
                    raw_probs = model.predict(preprocessed[None, ...], verbose=0)[0]

                    # Exponential Moving Average smoothing
                    if smoothed_probs is None:
                        smoothed_probs = raw_probs
                    else:
                        smoothed_probs = EMA_ALPHA * raw_probs + (1.0 - EMA_ALPHA) * smoothed_probs

                    # Evaluate top predictions
                    top3_indices = np.argsort(smoothed_probs)[-3:][::-1]
                    idx = int(top3_indices[0])
                    current_conf = float(smoothed_probs[idx])
                    margin = current_conf - float(smoothed_probs[top3_indices[1]])

                    top3_text = "  ".join(
                        f"{CLASSES[i]}:{smoothed_probs[i]:.0%}" for i in top3_indices
                    )

                    if current_conf >= confidence_threshold and margin >= MARGIN_THRESHOLD:
                        history.append(idx)
                    else:
                        history.append(-1)
                else:
                    history.append(-1)
            else:
                # No hand present: reset smoothing and candidate tracking
                history.append(-1)
                smoothed_probs = None
                display_letter = "No Hand Detected"

            # Determine stable majority vote from history
            valid_votes = [x for x in history if x >= 0]
            if valid_votes and hand_present:
                stable_idx, vote_count = Counter(valid_votes).most_common(1)[0]
                stable_letter = CLASSES[stable_idx]
                display_letter = f"{stable_letter} ({current_conf:.0%})"

                # Hold-to-confirm progress logic
                if stable_idx == candidate:
                    candidate_frames += 1
                else:
                    candidate = stable_idx
                    candidate_frames = 1

                # Confirm when gesture held stably for CONFIRMATION_FRAMES
                if candidate_frames >= CONFIRMATION_FRAMES:
                    if candidate != last_confirmed:
                        text_buffer += CLASSES[candidate]
                        last_confirmed = candidate
                        flash_frames = 12  # Visual confirmation flash
            else:
                candidate = None
                candidate_frames = 0
                if not hand_present:
                    last_confirmed = None
                if hand_present:
                    display_letter = "Uncertain"

            # Draw Guided ROI Box with corner accents
            box_color = (0, 255, 0) if hand_present else (0, 165, 255)
            draw_styled_rect(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), box_color, thickness=2)
            roi_label = "Hand Detection Zone" if hand_present else "Place Hand Here"
            cv2.putText(
                frame, roi_label, (roi_x1 + 8, roi_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1, cv2.LINE_AA
            )

            # Inset thumbnail preview of model input in top right corner
            if model_input_preview is not None:
                pw, ph = 110, 110
                px1 = roi_x2 - pw
                py1 = roi_y1 + 10
                frame[py1:py1+ph, px1:px1+pw] = model_input_preview
                cv2.rectangle(frame, (px1, py1), (px1+pw, py1+ph), (255, 255, 255), 1)
                cv2.putText(
                    frame, "Model Input", (px1 + 4, py1 + ph - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
                )

            # === HUD / Interface Rendering ===
            # Top Banner Background
            cv2.rectangle(frame, (0, 0), (roi_x1 - 10, 140), (25, 25, 25), -1)
            cv2.rectangle(frame, (0, 0), (roi_x1 - 10, 140), (70, 70, 70), 1)

            # Prediction & Status
            pred_color = (0, 255, 0) if hand_present and candidate is not None else (200, 200, 200)
            cv2.putText(
                frame, f"Prediction: {display_letter}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, pred_color, 2, cv2.LINE_AA
            )

            # Top 3 Candidates
            cv2.putText(
                frame, f"Top 3: {top3_text if top3_text else 'Waiting for gesture...'}",
                (15, 68), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 220, 0), 1, cv2.LINE_AA
            )

            # Hold-to-Confirm Progress Bar
            progress = min(1.0, candidate_frames / float(CONFIRMATION_FRAMES)) if candidate is not None else 0.0
            bar_x, bar_y, bar_w, bar_h = 15, 88, 220, 14
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * progress)
            bar_color = (0, 255, 0) if progress >= 1.0 else (0, 215, 255)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
            cv2.putText(
                frame, f"Hold: {int(progress * 100)}%",
                (bar_x + bar_w + 10, bar_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA
            )

            # Threshold indicator
            cv2.putText(
                frame, f"Thresh: {confidence_threshold:.0%}",
                (15, 126), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (160, 160, 160), 1, cv2.LINE_AA
            )

            # Bottom Text Buffer Banner
            cv2.rectangle(frame, (0, h - 85), (w, h), (20, 20, 20), -1)
            cv2.line(frame, (0, h - 85), (w, h - 85), (60, 60, 60), 1)

            # Confirmation flash banner
            if flash_frames > 0:
                cv2.rectangle(frame, (0, h - 85), (w, h), (0, 120, 0), 2)
                flash_frames -= 1

            # Display confirmed text buffer
            display_buffer = text_buffer if text_buffer else "[Empty]"
            buffer_color = (255, 255, 255) if text_buffer else (120, 120, 120)
            cv2.putText(
                frame, f"Text: {display_buffer}_",
                (15, h - 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, buffer_color, 2, cv2.LINE_AA
            )

            # Control Help
            controls_text = "Q: quit | C: clear | B: backspace | Space: space | +/-: threshold"
            cv2.putText(
                frame, controls_text,
                (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (180, 180, 180), 1, cv2.LINE_AA
            )

            cv2.imshow("Indian Sign Language Recognition (MobileNetV2)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # 'q' or ESC
                break
            elif key == ord("c"):
                text_buffer = ""
                candidate = None
                candidate_frames = 0
                last_confirmed = None
            elif key == ord("b") or key == 8:  # 'b' or Backspace
                text_buffer = text_buffer[:-1]
                last_confirmed = None
            elif key == 32:  # Space
                text_buffer += " "
                last_confirmed = None
            elif key == ord("+") or key == ord("="):
                confidence_threshold = min(0.95, round(confidence_threshold + 0.05, 2))
                print(f"Confidence threshold set to: {confidence_threshold:.0%}")
            elif key == ord("-") or key == ord("_"):
                confidence_threshold = max(0.40, round(confidence_threshold - 0.05, 2))
                print(f"Confidence threshold set to: {confidence_threshold:.0%}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam session terminated.")
        if text_buffer:
            print(f"Final transcribed text: \"{text_buffer}\"")


if __name__ == "__main__":
    main()
