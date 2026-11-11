"""Record landmark samples.  Run:  python collect_data.py [--session NAME]

Keys: 1-9 pick sign | SPACE start/stop recording | Q quit
Only frames with a detected hand are saved.
Record each sign in SEVERAL separate sessions (different day / lighting / distance / person).
"""
import argparse, csv, os, time
from collections import Counter
import cv2
import mediapipe as mp
from feature_utils import LABELS, RAW_DIM, extract_raw

CSV_PATH = os.path.join("data", "landmarks.csv")
HEADER = ["label", "session_id", "timestamp"] + [f"f{i}" for i in range(RAW_DIM)]


def existing_counts():
    c = Counter()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    c[row[0]] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=time.strftime("session_%Y%m%d_%H%M%S"))
    args = ap.parse_args()
    os.makedirs("data", exist_ok=True)
    new_file = not os.path.exists(CSV_PATH)
    counts = existing_counts()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: cannot open webcam (index 0). Close other apps using the camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=1,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    drawer, styles = mp.solutions.drawing_utils, mp.solutions.hands
    idx, recording, session_n = 0, False, 0

    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: camera read failed.")
                break
            frame = cv2.flip(frame, 1)  # mirror: SAME as app.py
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw, n = extract_raw(res)
            label = LABELS[idx]
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    drawer.draw_landmarks(frame, lm, styles.HAND_CONNECTIONS)
            if recording and n > 0:
                w.writerow([label, args.session, f"{time.time():.3f}"] + [f"{v:.6f}" for v in raw])
                counts[label] += 1
                session_n += 1
            col = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(frame, f"Sign: {label.upper()}  [{'REC' if recording else 'idle'}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(frame, f"samples (total): {counts[label]}  this run: {session_n}  hands: {n}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"session: {args.session}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "1-9 sign | SPACE rec | Q quit", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("SignBridge - collect", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            if k == 32:
                recording = not recording
            if ord("1") <= k <= ord("9"):
                idx = k - ord("1")
                recording = False
    cap.release()
    cv2.destroyAllWindows()
    print("Saved to", CSV_PATH, "| counts:", dict(counts))


if __name__ == "__main__":
    main()
