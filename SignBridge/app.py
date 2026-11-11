"""SignBridge live demo.

Run:
    python app.py

Controls:
    C = clear sequence
    S = speak sentence
    BACKSPACE = undo last sign
    Q = quit

Signs are automatically added after being held steadily for 1 second.
"""

import os
import sys
import threading
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np
import mediapipe as mp

from feature_utils import FEATURE_DIM, extract_raw, raw_to_features
from phrases import build_sentence


MODEL_PATH = os.path.join("models", "sign_model.pkl")

# Recognition settings
CONF_THRESHOLD = 0.70
WINDOW = 12
MIN_VOTES = 8
RELEASE_FRAMES = 10

# Hands-free confirmation
AUTO_CONFIRM_SECONDS = 1.0

CAM_W, CAM_H, PANEL_W = 800, 600, 440

BG, CARD = (28, 24, 22), (48, 42, 38)
WHITE, GREY, ACCENT = (240, 240, 240), (150, 150, 150), (230, 170, 60)
GREEN, AMBER, RED = (110, 200, 90), (60, 180, 240), (60, 60, 230)


class Speaker:
    def __init__(self):
        self.lock = threading.Lock()

    def speak(self, text):
        if text:
            threading.Thread(
                target=self._run,
                args=(text,),
                daemon=True
            ).start()

    def _run(self, text):
        if not self.lock.acquire(blocking=False):
            return

        try:
            import pyttsx3

            e = pyttsx3.init()
            e.setProperty("rate", 165)
            e.say(text)
            e.runAndWait()
            e.stop()

        except Exception as ex:
            print("Speech error:", ex)

        finally:
            self.lock.release()


def wrap(text, max_chars):
    lines = []
    cur = ""

    for w in text.split():
        if len(cur) + len(w) + 1 > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()

    if cur:
        lines.append(cur)

    return lines


def txt(img, s, org, scale=0.6, col=WHITE, th=1):
    cv2.putText(
        img,
        s,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        col,
        th,
        cv2.LINE_AA
    )


def render(cam, st):
    panel = np.full(
        (CAM_H, PANEL_W, 3),
        BG,
        np.uint8
    )

    # ---------------- HEADER ----------------
    txt(
        panel,
        "SignBridge",
        (20, 38),
        1.0,
        ACCENT,
        2
    )

    txt(
        panel,
        "ISL selected-vocabulary communication aid",
        (20, 60),
        0.45,
        GREY
    )

    # ---------------- DETECTED CARD ----------------
    cv2.rectangle(
        panel,
        (15, 75),
        (PANEL_W - 15, 200),
        CARD,
        -1
    )

    txt(
        panel,
        "DETECTED SIGN",
        (28, 98),
        0.45,
        GREY
    )

    shown = st["shown"]

    txt(
        panel,
        shown.upper() if shown else "--",
        (28, 138),
        1.2,
        WHITE if shown else GREY,
        2
    )

    bx, by, bw = 28, 158, PANEL_W - 56

    cv2.rectangle(
        panel,
        (bx, by),
        (bx + bw, by + 14),
        (70, 64, 60),
        -1
    )

    conf = st["conf"]

    cv2.rectangle(
        panel,
        (bx, by),
        (bx + int(bw * conf), by + 14),
        GREEN if conf >= CONF_THRESHOLD else AMBER,
        -1
    )

    tx = bx + int(bw * CONF_THRESHOLD)

    cv2.line(
        panel,
        (tx, by - 3),
        (tx, by + 17),
        WHITE,
        1
    )

    txt(
        panel,
        f"confidence {conf * 100:.0f}%  "
        f"(min {CONF_THRESHOLD * 100:.0f}%)",
        (28, 190),
        0.45,
        GREY
    )

    status, scol = st["status"]

    status_width = cv2.getTextSize(
        status,
        0,
        0.5,
        1
    )[0][0]

    txt(
        panel,
        status,
        (
            PANEL_W - 15 - 14 - status_width,
            98
        ),
        0.5,
        scol,
        1
    )

    # ---------------- SEQUENCE ----------------
    txt(
        panel,
        "CONFIRMED SEQUENCE",
        (20, 228),
        0.45,
        GREY
    )

    x, y = 15, 240

    for s in st["seq"]:
        w = cv2.getTextSize(
            s.upper(),
            0,
            0.6,
            2
        )[0][0] + 24

        if x + w > PANEL_W - 10:
            x, y = 15, y + 40

        cv2.rectangle(
            panel,
            (x, y),
            (x + w, y + 32),
            ACCENT,
            -1
        )

        txt(
            panel,
            s.upper(),
            (x + 12, y + 22),
            0.6,
            (20, 20, 20),
            2
        )

        x += w + 8

    if not st["seq"]:
        txt(
            panel,
            "Hold a sign for 1 second to add it",
            (20, 262),
            0.5,
            GREY
        )

    # ---------------- SENTENCE ----------------
    emerg = "emergency" in st["sentence"].lower()

    cv2.rectangle(
        panel,
        (15, 350),
        (PANEL_W - 15, 500),
        (40, 40, 120) if emerg else CARD,
        -1
    )

    txt(
        panel,
        "SENTENCE",
        (28, 372),
        0.45,
        GREY
    )

    sentence_lines = (
        wrap(st["sentence"], 28)[:3]
        or ["--"]
    )

    for i, ln in enumerate(sentence_lines):
        txt(
            panel,
            ln,
            (28, 408 + i * 30),
            0.75,
            WHITE,
            2
        )

    if st["speaking_flash"] > time.time():
        txt(
            panel,
            "speaking...",
            (PANEL_W - 130, 372),
            0.5,
            GREEN
        )

    # ---------------- CONTROLS ----------------
    txt(
        panel,
        "AUTO ADD | C clear | S speak | BKSP undo | Q quit",
        (15, 530),
        0.45,
        GREY
    )

    txt(
        panel,
        "AI communication aid - limited ISL vocabulary.",
        (15, 556),
        0.4,
        GREY
    )

    txt(
        panel,
        "Not full translation. Not medical advice. Not an interpreter.",
        (15, 574),
        0.4,
        GREY
    )

    # ---------------- EMERGENCY VISUAL ----------------
    if emerg:

        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(
                cam,
                (0, 0),
                (CAM_W, 54),
                (0, 0, 200),
                -1
            )

            txt(
                cam,
                "EMERGENCY",
                (CAM_W // 2 - 110, 38),
                1.2,
                WHITE,
                3
            )

        cv2.rectangle(
            cam,
            (0, 0),
            (CAM_W - 1, CAM_H - 1),
            (0, 0, 220),
            6
        )

    return np.hstack([cam, panel])


def main():

    # ---------------- MODEL CHECK ----------------
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            "MODEL MISSING: models/sign_model.pkl not found.\n"
            "1) python collect_data.py\n"
            "2) python train.py\n"
            "Then run python app.py again."
        )

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    classes = list(bundle["labels"])

    if bundle.get("feature_dim") != FEATURE_DIM:
        sys.exit(
            "Model/feature mismatch. "
            "Re-run python train.py with the current code."
        )

    # ---------------- CAMERA ----------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        sys.exit(
            "ERROR: cannot open webcam. "
            "Close other apps using it."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # ---------------- MEDIAPIPE ----------------
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    drawer = mp.solutions.drawing_utils

    speaker = Speaker()

    # ---------------- STATE ----------------
    votes = deque(maxlen=WINDOW)

    seq = []

    shown = None
    conf = 0.0

    no_hand = 0

    last_added = None

    # Time at which current stable sign started
    stable_since = None

    flash = 0.0

    cv2.namedWindow(
        "SignBridge",
        cv2.WINDOW_AUTOSIZE
    )

    # ---------------- MAIN LOOP ----------------
    while True:

        ok, frame = cap.read()

        if not ok:
            print("Camera read failed.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # MediaPipe
        res = hands.process(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
        )

        raw, n = extract_raw(res)

        cam = cv2.resize(
            frame,
            (CAM_W, CAM_H)
        )

        # Draw hand landmarks
        if res.multi_hand_landmarks:

            for lm in res.multi_hand_landmarks:

                drawer.draw_landmarks(
                    cam,
                    lm,
                    mp_hands.HAND_CONNECTIONS
                )

        stable = None

        # ==================================================
        # NO HAND
        # ==================================================
        if n == 0:

            no_hand += 1

            conf = 0.0

            stable_since = None

            if no_hand >= RELEASE_FRAMES:

                votes.clear()

                shown = None

            status = (
                "No hand",
                GREY
            )

        # ==================================================
        # HAND DETECTED
        # ==================================================
        else:

            no_hand = 0

            # Predict
            features = raw_to_features(raw).reshape(
                1,
                -1
            )

            proba = model.predict_proba(features)[0]

            i = int(
                np.argmax(proba)
            )

            conf = float(
                proba[i]
            )

            # Add prediction to voting window
            if conf >= CONF_THRESHOLD:

                votes.append(
                    (
                        classes[i],
                        conf
                    )
                )

            else:

                votes.append(
                    (
                        None,
                        conf
                    )
                )

            # Count votes
            c = Counter(
                v[0]
                for v in votes
                if v[0]
            )

            if c:

                lab, cnt = c.most_common(1)[0]

                if cnt >= MIN_VOTES:

                    stable = lab

            # ==================================================
            # STABLE SIGN
            # ==================================================
            if stable:

                shown = stable

                # New stable sign detected
                if stable_since is None:

                    stable_since = time.time()

                held_for = (
                    time.time()
                    - stable_since
                )

                # ----------------------------------------------
                # Automatically confirm after 1 second
                # ----------------------------------------------
                if (
                    held_for >= AUTO_CONFIRM_SECONDS
                    and stable != last_added
                ):

                    seq.append(stable)

                    last_added = stable

                    stable_since = None

                    votes.clear()

                    status = (
                        "ADDED",
                        GREEN
                    )

                else:

                    remaining = max(
                        0.0,
                        AUTO_CONFIRM_SECONDS - held_for
                    )

                    status = (
                        f"HOLD {remaining:.1f}s",
                        GREEN
                    )

            # ==================================================
            # UNCERTAIN
            # ==================================================
            else:

                stable_since = None

                status = (
                    "Uncertain...",
                    AMBER
                )

        # ==================================================
        # BUILD SENTENCE
        # ==================================================
        sentence = build_sentence(seq)

        # ==================================================
        # KEYBOARD INPUT
        # ==================================================
        k = cv2.waitKey(1) & 0xFF

        # Quit
        if k in (
            ord("q"),
            ord("Q")
        ):

            break

        # Clear
        elif k in (
            ord("c"),
            ord("C")
        ):

            seq.clear()

            last_added = None

            stable_since = None

            votes.clear()

            sentence = ""

        # Undo
        elif k == 8 and seq:

            seq.pop()

            last_added = (
                seq[-1]
                if seq
                else None
            )

            stable_since = None

            votes.clear()

            sentence = build_sentence(seq)

        # Speak
        elif k in (
            ord("s"),
            ord("S")
        ):

            if sentence:

                speaker.speak(
                    sentence
                )

                flash = (
                    time.time()
                    + 2.0
                )

        # ==================================================
        # DISPLAY
        # ==================================================
        cv2.imshow(
            "SignBridge",
            render(
                cam,
                {
                    "shown": shown,
                    "conf": conf,
                    "status": status,
                    "seq": seq,
                    "sentence": sentence,
                    "speaking_flash": flash
                }
            )
        )

    # ---------------- CLEANUP ----------------
    cap.release()

    hands.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()