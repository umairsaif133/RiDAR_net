import numpy as np

# -----------------------------
# Normalization utilities
# -----------------------------
def normalize_xyz(xyz):
    centroid = np.mean(xyz, axis=0, keepdims=True)
    xyz_centered = xyz - centroid
    max_dist = np.max(np.linalg.norm(xyz_centered, axis=1, keepdims=True))
    if max_dist < 1e-6:
        max_dist = 1.0
    return xyz_centered / max_dist


def scale_feature(arr):
    mi, ma = arr.min(), arr.max()
    if ma - mi < 1e-6:
        return np.ones_like(arr) * 0.5
    normalized = (arr - mi) / (ma - mi)
    return normalized * 0.9 + 0.1


# -----------------------------
# Sampling utilities
# -----------------------------
def random_sample(points, target_n):
    n = len(points)
    if n >= target_n:
        idx = np.random.choice(n, target_n, replace=False)
        return points[idx]
    else:
        pad_size = target_n - n
        pad = np.zeros((pad_size, points.shape[1]))
        return np.vstack([points, pad])