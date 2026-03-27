import os
import re
import torch
import numpy as np
import pandas as pd

from utils import normalize_xyz, scale_feature, random_sample


class LiDARDatasetProcessor:
    def __init__(
        self,
        csv_file,
        max_points=512,
        min_points=10,
        remove_outliers=False,
        clip_percentile=0.01,
    ):
        self.csv_file = csv_file
        self.max_points = max_points
        self.min_points = min_points
        self.remove_outliers = remove_outliers
        self.clip_percentile = clip_percentile

    def load_csv(self):
        print(f"Loading {self.csv_file}...")
        df = pd.read_csv(self.csv_file, low_memory=False)
        print(f"Loaded dataset: {len(df)} rows")
        return df

    def get_feature_columns(self, df):
        feature_cols = [c for c in df.columns if re.match(r"^[xyzie]\d+$", c)]

        def feature_key(c):
            name = c[0]
            idx = int(c[1:])
            order = {"x": 0, "y": 1, "z": 2, "i": 3, "e": 4}
            return (idx, order[name])

        feature_cols = sorted(feature_cols, key=feature_key)
        return feature_cols

    def process_cluster(self, flat_features):
        n_points = len(flat_features) // 5
        features = flat_features.reshape((n_points, 5))

        # remove zero rows
        mask = np.any(features != 0, axis=1)
        features = features[mask]

        if len(features) < self.min_points:
            return None

        xyz = features[:, :3]
        intensity = features[:, 3:4]
        extra = features[:, 4:5]

        # normalize
        xyz = normalize_xyz(xyz)
        intensity = scale_feature(intensity)
        extra = scale_feature(extra)

        cluster = np.concatenate([xyz, intensity, extra], axis=1)

        # resample
        cluster = random_sample(cluster, self.max_points)

        return cluster.astype(np.float32)

    def run(self):
        df = self.load_csv()
        feature_cols = self.get_feature_columns(df)

        clusters = df.groupby("cluster_id")

        X, y, splits, ids = [], [], [], []

        for cid, group in clusters:
            flat = group[feature_cols].values[0].astype(np.float32)

            processed = self.process_cluster(flat)
            if processed is None:
                continue

            X.append(processed)
            y.append(group["label"].iloc[0])
            splits.append(group["split"].iloc[0])
            ids.append(cid)

        return X, y, splits, ids


def save_pt_splits(X, y, splits, ids, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    def split_data(name):
        idx = [i for i, s in enumerate(splits) if s == name]
        Xs = [X[i] for i in idx]
        ys = [y[i] for i in idx]
        ids_s = [ids[i] for i in idx]

        if len(Xs) == 0:
            return torch.empty(0), torch.empty(0), []

        return (
            torch.tensor(np.stack(Xs), dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long),
            ids_s,
        )

    train = split_data("train")
    val = split_data("val")
    test = split_data("test")

    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val, os.path.join(out_dir, "val.pt"))
    torch.save(test, os.path.join(out_dir, "test.pt"))

    print("Saved processed datasets to:", out_dir)