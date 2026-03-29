import numpy as np
from scipy.ndimage import label

image = np.load('stars.npy')
labeled, _ = label(image)

clean = np.zeros_like(image)
for i in range(1, labeled.max() + 1):
    obj = labeled == i
    ys, xs = np.where(obj)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    if h == 3 and w == 3:
        continue
    if h == 1 and w == 1:
        continue
    clean[obj] = image[obj]

_, num_stars = label(clean.astype(bool))
print(f'{num_stars} звездочек')
