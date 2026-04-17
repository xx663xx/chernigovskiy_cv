import numpy as np
from scipy.ndimage import binary_hit_or_miss

img = np.load("stars.npy").astype(bool)

plus = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=bool)

cross = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
], dtype=bool)

plus_n = binary_hit_or_miss(img, structure1=plus, structure2=~plus).sum()
x_n = binary_hit_or_miss(img, structure1=cross, structure2=~cross).sum()

print("Плюсов:", int(plus_n))
print("Крестов:", int(x_n))
print("Всего:", int(plus_n + x_n))