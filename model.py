
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def window(event=True, length=200, noise_level=0.05):
   
    t = np.linspace(0, 1, length)

    base = 0.02 * np.sin(2 * np.pi * 1.0 * t)
    if event:
        burst = np.zeros_like(t)
        center = np.random.uniform(0.3, 0.7)
        width = np.random.uniform(0.05, 0.12)
        burst_idx = np.exp(-((t - center) ** 2) / (2 * width ** 2))

        ax = 0.5 * burst_idx * (np.sin(2 * np.pi * 5 * (t - center)) + 1)
        ay = 0.4 * burst_idx * (np.cos(2 * np.pi * 6 * (t - center)) + 1)
        az = 0.6 * burst_idx * (np.sin(2 * np.pi * 4 * (t - center)) + 1)
        x = base + ax + noise_level * np.random.randn(length)
        y = base + ay + noise_level * np.random.randn(length)
        z = base + az + noise_level * np.random.randn(length)
    else:
        x = base + 0.01 * np.random.randn(length)
        y = base + 0.01 * np.random.randn(length)
        z = base + 0.01 * np.random.randn(length)
    return np.vstack([x, y, z]).T  

def info(window):
    # mean std max-min KE zero crossing count
    feats = []
    for ax in range(3):
        v = window[:, ax]
        feats.append(np.mean(v))
        feats.append(np.std(v))
        feats.append(np.max(v) - np.min(v))
        feats.append(np.sum(v**2) / len(v))  
        zc = ((v[:-1] * v[1:]) < 0).sum()
        feats.append(zc)

    mag = np.linalg.norm(window, axis=1)
    feats.append(np.mean(mag))
    feats.append(np.std(mag))
    feats.append(np.max(mag))
    return np.array(feats)

def fin(n_samples=600, length=200):
    X = []
    y = []
    rows = []
    for i in range(n_samples):
        if i < n_samples // 2:
            w = window(event=True, length=length)
            label = 1
        else:
            w = window(event=False, length=length)
            label = 0
        feats = info(w)
        X.append(feats)
        y.append(label)

        row = np.concatenate([feats, [label]])
        rows.append(row)
    X = np.vstack(X)
    y = np.array(y)
    cols = [f"f{i}" for i in range(X.shape[1])] + ["label"]
    df = pd.DataFrame(np.hstack([X, y.reshape(-1,1)]), columns=cols)
    return X, y, df

def main():
    out_dir = "micro_event_demo_out"
    os.makedirs(out_dir, exist_ok=True)

    print("Building dataset")
    X, y, df = fin(n_samples=600, length=200)
    csv_path = os.path.join(out_dir, "sample_sensor_data.csv")
    df.sample(frac=1, random_state=RANDOM_SEED).to_csv(csv_path, index=False)
    print(f"Saved sample data to {csv_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training RandomForest classifier")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    model_path = os.path.join(out_dir, "rf_model.pkl")
    joblib.dump({"model": clf, "scaler": scaler}, model_path)
    print(f"Saved model+scaler to {model_path}")

if __name__ == "__main__":
    main()
