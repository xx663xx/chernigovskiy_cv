import numpy as np
from pathlib import Path
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt

out_dir = Path(__file__).resolve().parent / 'out'
files = sorted(out_dir.glob('h_*.npy'), key=lambda p: int(p.stem.split('_')[1]))

def get_centroids(mask):
    lbl, n = label(mask.astype(bool))
    centroids = []
    for i in range(1, n + 1):
        cy, cx = center_of_mass(mask, lbl, i)
        centroids.append((cy, cx))
    return np.array(centroids) if centroids else np.array([]).reshape(0, 2)

def match_centroids(prev, curr):
    if len(prev) == 0 or len(curr) == 0:
        return curr
    matched = []
    used = set()
    for p in prev:
        best_idx, best_dist = -1, float('inf')
        for j, c in enumerate(curr):
            if j not in used:
                d = np.sum((p - c) ** 2)
                if d < best_dist:
                    best_dist, best_idx = d, j
        if best_idx >= 0:
            matched.append(curr[best_idx])
            used.add(best_idx)
    for j, c in enumerate(curr):
        if j not in used:
            matched.append(c)
    return np.array(matched)

paths = []
prev = get_centroids(np.load(files[0]))
paths = [[tuple(p)] for p in prev]

for f in files[1:]:
    curr = get_centroids(np.load(f))
    matched = match_centroids(prev, curr)
    for i, p in enumerate(matched):
        if i < len(paths):
            paths[i].append(tuple(p))
        else:
            paths.append([tuple(p)])
    prev = matched

plt.figure(figsize=(10, 8))
for i, path in enumerate(paths):
    coords = np.array(path)
    plt.plot(coords[:, 1], coords[:, 0], '-o', markersize=3, label=f'Объект {i+1}')

plt.gca().invert_yaxis()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Траектории движения объектов')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
