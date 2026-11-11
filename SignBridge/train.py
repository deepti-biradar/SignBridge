"""Train a Random Forest on collected landmarks.  Run:  python train.py [--final]
Session-aware split when >=2 sessions exist; otherwise stratified split with a loud warning.
--final refits the saved model on ALL data after reporting held-out metrics.
"""
import argparse, os, sys
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from feature_utils import FEATURE_DIM, LABELS, RAW_DIM, raw_matrix_to_features

CSV_PATH = os.path.join("data", "landmarks.csv")
MODEL_PATH = os.path.join("models", "sign_model.pkl")


def make_split(X, y, groups):
    if groups is not None and len(set(groups)) >= 2:
        gss = GroupShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
        for tr, te in gss.split(X, y, groups):
            if set(y[tr]) == set(y) and len(te) > 0:
                print(f"Session-aware split: train sessions={len(set(groups[tr]))}, "
                      f"test sessions={len(set(groups[te]))}")
                return tr, te, True
        print("WARNING: no session split keeps every sign in training. Falling back to stratified.")
    else:
        print("WARNING: fewer than 2 sessions (or no session metadata). Using a STRATIFIED random split.")
    print("WARNING: frames from the same recording land in train AND test, so metrics will be "
          "optimistic. Record each sign in multiple separate sessions.")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=42)
    return tr, te, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(CSV_PATH):
        sys.exit(f"ERROR: {CSV_PATH} not found. Run collect_data.py first.")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["label"])
    cols = [f"f{i}" for i in range(RAW_DIM)]
    if not all(c in df.columns for c in cols):
        sys.exit("ERROR: CSV does not contain the 126 raw feature columns.")
    y = df["label"].astype(str).values
    counts = pd.Series(y).value_counts()
    print("Samples per sign:\n", counts.to_string())
    missing = [l for l in LABELS if l not in counts.index]
    if missing:
        print("WARNING: no data for:", missing)
    if len(counts) < 2:
        sys.exit("ERROR: need at least 2 different signs to train.")
    if counts.min() < 50:
        print("WARNING: some signs have <50 samples; results will be unreliable.")

    groups = df["session_id"].astype(str).values if "session_id" in df.columns and df["session_id"].notna().all() else None
    X = raw_matrix_to_features(df[cols].values.astype(np.float32))
    assert X.shape[1] == FEATURE_DIM
    tr, te, session_aware = make_split(X, y, groups)

    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])

    labels = sorted(set(y))
    print(f"\nHeld-out accuracy: {accuracy_score(y[te], pred):.3f}  "
          f"({'session-aware' if session_aware else 'STRATIFIED - optimistic'})")
    p, r, f, _ = precision_recall_fscore_support(y[te], pred, average="macro", zero_division=0)
    print(f"Macro precision: {p:.3f}  recall: {r:.3f}  F1: {f:.3f}\n")
    print(classification_report(y[te], pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):\nlabels:", labels)
    print(confusion_matrix(y[te], pred, labels=labels))
    print("\nNOTE: offline frame-level metrics do NOT prove real-world performance. "
          "Evaluate with unseen signers and separate sessions.")

    if args.final:
        print("Refitting on ALL data for the saved model...")
        model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "labels": list(model.classes_), "feature_dim": FEATURE_DIM}, MODEL_PATH)
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()
