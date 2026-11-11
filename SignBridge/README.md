# SignBridge - ISL selected-vocabulary communication aid (ideathon prototype)

Camera -> MediaPipe hand landmarks -> stable sign -> confirmed sequence -> phrase -> speech.
Vocabulary (9 signs): help, water, food, doctor, hospital, emergency, hurt, disagree, urgent.

## Positioning (read this before presenting)
- NOT full ISL translation. Only 9 isolated signs, performed the way *your* team signs them.
- NOT medical diagnosis. NOT a replacement for human interpreters.
- An AI communication aid. Emergency is one high-impact demo context, not the whole product.
- Fully offline. No cloud APIs, no keys. No dataset or model is shipped - you record your own.

## Setup (Windows PowerShell, from the SignBridge folder)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`

## 1. Collect data
```powershell
python collect_data.py --session s1
```
Keys 1-9 pick sign (order above), SPACE start/stop recording, Q quit.
Hold each sign ~15-20 s while moving slightly (distance, angle). Target 300+ frames per sign per session.
Repeat with `--session s2`, `--session s3` (different time, lighting, distance, ideally another person).
Data is appended to `data/landmarks.csv` (label, session_id, timestamp, 126 raw landmark values).

## 2. Train
```powershell
python train.py --final
```
Uses a session-aware split when >=2 sessions exist (else a stratified split with a warning).
Prints accuracy, precision, recall, F1, confusion matrix. Saves `models/sign_model.pkl`.
`--final` refits on all data after reporting held-out metrics.

## 3. Run the demo
```powershell
python app.py
```
| Key | Action |
|---|---|
| SPACE | add/confirm the currently STABLE sign |
| C | clear sequence |
| S | speak sentence |
| BACKSPACE | undo last sign |
| Q | quit |

Emergency demo: show HELP (wait for "STABLE", press SPACE), lower hand, show EMERGENCY (SPACE), press S.
Expected: "I need help. This is an emergency." with a flashing red banner.

## How stability works
- Predictions below 70% confidence count as "uncertain" and never change the display.
- A sign is stable only if it wins >=8 of the last 12 frames.
- A held sign cannot be added twice: lower your hand (or change sign) before adding again.
- Nothing enters the sequence without you pressing SPACE.
Tune `CONF_THRESHOLD`, `WINDOW`, `MIN_VOTES` at the top of `app.py`.

## Design notes
- `feature_utils.py` is the single feature transform for collector, trainer and app (131 features: per-hand
  wrist-centred, scale-normalised landmarks + inter-hand offset + presence flags).
- Frames are mirrored before MediaPipe in BOTH collector and app so handedness is consistent.
- Hand order is deterministic: slot 0 = Left, slot 1 = Right, fallback to wrist x.

## Real evaluation (important)
Frame-level accuracy from `train.py` does NOT prove real-world performance. Consecutive frames are
near-duplicates, so even a session-aware split from one person in one room is optimistic.
A credible evaluation needs:
- multiple signers (including ones not in training),
- separate recording sessions (different days, lighting, backgrounds, camera distance),
- held-out signers/sessions only for testing,
- sign-level (not frame-level) success rate in live use, with false-trigger counts.
Report it honestly as a prototype on a limited vocabulary.

## Troubleshooting
- Camera won't open: close Zoom/Teams/browser tabs; change index 0 to 1 in the scripts.
- Two signs confused: record more varied samples; make signs more visually distinct.
- No speech: check Windows volume/default voice (Settings > Time & language > Speech).
