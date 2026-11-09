import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import subprocess
from pathlib import Path

from sklearn.model_selection import train_test_split

repo_dir = Path("/content/replicating_RePO")
os.chdir(repo_dir)
if not (repo_dir / "data").exists():
    print("Extraindo dados...")
    subprocess.run(["unzip", "data/new_data/data/new_data/CIC-ToN-IoT-Dataset-Raw.zip", "-d", "data/new_data/"], check=True)
else:
    print("Dados já extraídos.")


print("Configuration")
DATASET_PATH = "./data/new_data/CIC-ToN-IoT.csv"
OUTPUT_DIR = "./data/new_data"

TARGET_COLUMN = "Label"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/x_train_meta", exist_ok=True)

print("Load dataset")
total_rows = sum(1 for _ in open(DATASET_PATH)) - 1

chunks = []
chunk_size = 100_000

with tqdm(total=total_rows, desc="Loading CSV") as pbar:
    for chunk in pd.read_csv(DATASET_PATH, chunksize=chunk_size):

        chunk.insert(
            0,
            "column00",
            np.arange(row_offset, row_offset + len(chunk))
        )

        row_offset += len(chunk)

        # Normalize BENIGN labels
        attack = chunk["Attack"].copy()
        is_benign = (
            attack.astype("string")
            .str.strip()
            .str.upper()
            .eq("BENIGN")
            .fillna(False)
        )

        labels = attack.copy()
        labels.loc[is_benign] = "BENIGN"

        chunk = chunk.drop(columns=["Label", "Attack"])
        chunk["Label"] = labels

        chunks.append(chunk)
        pbar.update(len(chunk))

df = pd.concat(chunks, ignore_index=True)
del chunks

print("Separate benign and malicious samples")
benign = df[df[TARGET_COLUMN].str.strip().str.upper() == "BENIGN"]
malicious = df[df[TARGET_COLUMN].str.strip().str.upper() != "BENIGN"]

print("Split benign samples: 70% training, 30% testing")
benign_train, benign_test = train_test_split(
    benign,
    test_size=0.30,
    random_state=42
)

print("Training contains only benign samples")
train = benign_train.copy()

print("Testing contains remaining benign + all malicious samples")
test = pd.concat(
    [benign_test, malicious],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

print("Separate features")
feats = train.iloc[:, 8:]
ds_port = train.iloc[:, 5]

train_features = pd.concat([ds_port, feats], axis=1)
X_train = train_features.iloc[:, :-1].astype(np.float32)

print("Remove invalid training samples for min/max calculation")
valid_train = np.isfinite(X_train.values).all(axis=1)
X_train = X_train.loc[valid_train]

train = train.loc[valid_train]

print("Calculate min and max exclusively from training data")
train_min = X_train.min(axis=0).values
train_max = X_train.max(axis=0).values

print("Save min and max")
np.save(f"{OUTPUT_DIR}/x_train_meta/train_min.npy", train_min)
np.save(f"{OUTPUT_DIR}/x_train_meta/train_max.npy", train_max)
print("\nTraining min/max saved successfully!")

print("Save original, unnormalized datasets")
train.to_csv(f"{OUTPUT_DIR}/new_train.csv", index=False)
test.to_csv(f"{OUTPUT_DIR}/new_test.csv", index=False)

print("Display information")
print("Training dataset:")
print(f"Samples: {len(train)}")
print(train[TARGET_COLUMN].value_counts())

print("\nTesting dataset:")
print(f"Samples: {len(test)}")
print(test[TARGET_COLUMN].value_counts())

