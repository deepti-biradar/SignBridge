"""Shared landmark extraction + feature transform.
Used by collect_data.py, train.py and app.py so training and inference match exactly.

Raw layout (126 values): slot 0 = Left hand, slot 1 = Right hand, each 21 landmarks x (x, y, z).
Missing hand -> zeros.
Engineered features (131 values):
  [0:126]   per-hand landmarks, wrist-centred and scaled by the hand's max wrist distance
  [126:129] wrist1 - wrist0 offset, scaled by mean hand size (zeros unless two hands)
  [129:131] hand-present flags
IMPORTANT: always feed MediaPipe the SAME orientation (we mirror the frame first, everywhere).
"""
import numpy as np

LABELS = ["help", "water", "food", "doctor", "hospital",
          "emergency", "hurt", "disagree", "urgent"]
NUM_LANDMARKS = 21
HAND_DIM = NUM_LANDMARKS * 3
RAW_DIM = 2 * HAND_DIM          # 126
FEATURE_DIM = RAW_DIM + 3 + 2   # 131


def extract_raw(results):
    """MediaPipe Hands result -> (raw[126] float32, number_of_hands)."""
    raw = np.zeros(RAW_DIM, dtype=np.float32)
    if not results or not results.multi_hand_landmarks:
        return raw, 0

    hands = []
    for i, lm in enumerate(results.multi_hand_landmarks):
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
        label = None
        if results.multi_handedness and i < len(results.multi_handedness):
            label = results.multi_handedness[i].classification[0].label  # "Left"/"Right"
        hands.append((label, pts))
    hands = hands[:2]

    slots = {}
    if len(hands) == 2 and hands[0][0] == hands[1][0]:
        # Same handedness label twice (misclassification): order by wrist x
        hands.sort(key=lambda h: h[1][0, 0])
        slots[0], slots[1] = hands[0][1], hands[1][1]
    else:
        for label, pts in hands:
            slot = 0 if label == "Left" else 1
            if slot in slots:  # safety
                slot = 1 - slot
            slots[slot] = pts

    for slot, pts in slots.items():
        raw[slot * HAND_DIM:(slot + 1) * HAND_DIM] = pts.reshape(-1)
    return raw, len(hands)


def raw_to_features(raw):
    """126 raw values -> 131 normalised features (translation + scale invariant)."""
    raw = np.asarray(raw, dtype=np.float32).reshape(RAW_DIM)
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    wrists, scales, present = [], [], [0.0, 0.0]

    for s in range(2):
        pts = raw[s * HAND_DIM:(s + 1) * HAND_DIM].reshape(NUM_LANDMARKS, 3)
        if not np.any(pts):
            wrists.append(None)
            scales.append(None)
            continue
        centred = pts - pts[0]
        scale = float(np.max(np.linalg.norm(centred, axis=1)))
        if scale < 1e-6:
            wrists.append(None)
            scales.append(None)
            continue
        feats[s * HAND_DIM:(s + 1) * HAND_DIM] = (centred / scale).reshape(-1)
        wrists.append(pts[0])
        scales.append(scale)
        present[s] = 1.0

    if wrists[0] is not None and wrists[1] is not None:
        mean_scale = (scales[0] + scales[1]) / 2.0
        feats[RAW_DIM:RAW_DIM + 3] = (wrists[1] - wrists[0]) / mean_scale
    feats[RAW_DIM + 3:RAW_DIM + 5] = present
    return feats


def raw_matrix_to_features(X):
    return np.vstack([raw_to_features(r) for r in X])
