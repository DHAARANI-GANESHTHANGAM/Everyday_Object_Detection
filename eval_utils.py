import os, datetime
import pandas as pd
from ultralytics import YOLO

PROJECT = "/content/drive/MyDrive/Everyday_Object_Detection"
DATA = f"{PROJECT}/data/dataset.yaml"
CLASSES = ["Glasses", "Sunglasses", "Headphones", "Watch", "Pen", "Hat"]
RESULTS_CSV = f"{PROJECT}/results/results.csv"
os.makedirs(f"{PROJECT}/results", exist_ok=True)


def train_version(version, base="yolov8n.pt", epochs=30, imgsz=640, batch=16, **kwargs):
    """Fine-tune a YOLO model and return the best checkpoint."""
    model = YOLO(base)
    model.train(data=DATA, epochs=epochs, imgsz=imgsz, batch=batch, seed=42,
                project=f"{PROJECT}/runs/train", name=version, exist_ok=True, **kwargs)
    return YOLO(f"{PROJECT}/runs/train/{version}/weights/best.pt")


def evaluate_and_log(model, version, change, imgsz=640, notes=""):
    """Validate a model on the val split and write one row to results.csv."""
    m = model.val(data=DATA, imgsz=imgsz, batch=16, split="val",
                  project=f"{PROJECT}/runs/val", name=version,
                  exist_ok=True, plots=True, verbose=False)
    ms = m.speed["preprocess"] + m.speed["inference"] + m.speed["postprocess"]
    row = {
        "version": version,
        "date": datetime.date.today().isoformat(),
        "change": change,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "ms_per_image": round(ms, 2),
        "FPS": round(1000 / ms, 1),
        "notes": notes,
    }
    per_class = dict(zip(m.box.ap_class_index.tolist(), m.box.ap50.tolist()))
    for i, c in enumerate(CLASSES):
        row[f"AP50_{c}"] = round(float(per_class.get(i, 0.0)), 4)

    df = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    if len(df):
        df = df[df["version"] != version]      # re-running a version replaces its row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"{version}: mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}  FPS={row['FPS']}")
    return row


def regression_check(new_version, overall_tol=0.01, class_tol=0.03):
    """Compare a version with the champion (best mAP50-95 so far).
    FAIL if overall mAP50 drops more than overall_tol, or any class AP50
    drops more than class_tol. Saves status and reasons into results.csv."""
    df = pd.read_csv(RESULTS_CSV)
    for col in ["status", "compared_to", "reasons"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    idx = df.index[df["version"] == new_version][-1]
    new = df.loc[idx]
    others = df[df["version"] != new_version]

    if others.empty:
        df.loc[idx, ["status", "compared_to", "reasons"]] = ["BASELINE", "", ""]
        df.to_csv(RESULTS_CSV, index=False)
        print(f"[BASELINE] {new_version} is the first version.")
        return "BASELINE", []

    champ = others.loc[others["mAP50_95"].idxmax()]
    reasons = []
    d = new["mAP50"] - champ["mAP50"]
    if d < -overall_tol:
        reasons.append(f"overall mAP50 {d:+.3f}")
    for c in CLASSES:
        d = new[f"AP50_{c}"] - champ[f"AP50_{c}"]
        if d < -class_tol:
            reasons.append(f"{c} AP50 {d:+.3f}")

    status = "FAIL" if reasons else "PASS"
    df.loc[idx, ["status", "compared_to", "reasons"]] = [status, champ["version"], "; ".join(reasons)]
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[{status}] {new_version} vs champion {champ['version']}")
    for r in reasons:
        print("   -", r)
    return status, reasons
