import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening
from pathlib import Path

path = Path('/Users/xx/Documents/2 semestr/CV/wires/wires')

struct = np.ones((3, 1))

for i in range(1, 7):
    image = np.load(path/f"wires{i}.npy")
    labeled = label(image)
    wires_count = np.max(labeled)
    print("")
    print(f"Image {i} has {wires_count} wires")

    for j in range(1, wires_count + 1):
        wire = labeled == j
        process = opening(wire, struct)
        labeled_process = label(process)
        parts = np.max(labeled_process)
        print(f"Wire {j} has {parts} parts after processing")

    plt.subplot(121)
    plt.title("Original")
    plt.imshow(image)
    plt.subplot(122)
    plt.title("Process")
    plt.imshow(opening(image, struct))
    plt.show()