import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.transform import resize

save_path = Path(__file__).parent

def make_binary(image, threshold, mode):
    if image.ndim == 3:
        image = image[:, :, :3].mean(axis=2)
    if mode == 'lt':
        return image < threshold
    return image > threshold

def sort_regions(regions):
    return sorted(regions, key = lambda region: (region.bbox[1], region.bbox[0]))

def count_holes(region_image):
    temp = np.zeros((region_image.shape[0] + 2, region_image.shape[1] + 2), dtype=bool)
    temp[1:-1, 1:-1] = region_image
    inv = np.logical_not(temp)
    labeled = label(inv)
    return np.max(labeled) - 1

def get_feature_vector(region_image):
    small = resize(region_image.astype(float), (20, 20), order=0, anti_aliasing=False, preserve_range=True)
    small = (small > 0.5).astype(float)
    h, w = region_image.shape
    area = region_image.sum() / region_image.size
    aspect = h / w
    holes = count_holes(region_image)
    row_sums = small.sum(axis=1) / 20
    col_sums = small.sum(axis=0) / 20
    vector = [area, aspect, holes]
    vector.extend(row_sums.tolist())
    vector.extend(col_sums.tolist())
    vector.extend(small.flatten().tolist())
    return np.array(vector, dtype = float)

def get_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

alphabet_names = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '/', '-']
alphabet_image = imread(save_path / 'img' / 'alphabet-small.png')
alphabet_binary = make_binary(alphabet_image, 200, 'lt')
alphabet_labeled = label(alphabet_binary)
alphabet_regions = [region for region in regionprops(alphabet_labeled) if region.area > 20]
alphabet_regions = sort_regions(alphabet_regions)
patterns = {}

for i in range(len(alphabet_regions)):
    symbol = alphabet_names[i]
    patterns[symbol] = get_feature_vector(alphabet_regions[i].image)

image = imread(save_path / 'img' / 'alphabet.png')
binary = make_binary(image, 1, 'gt')
labeled = label(binary)
regions = [region for region in regionprops(labeled) if region.area > 20]
result = {}
out_path = save_path / 'out_vector'
out_path.mkdir(exist_ok = True)
plt.figure(figsize=(5, 7))

for region in regions:
    vector = get_feature_vector(region.image)
    best_symbol = ''
    best_distance = 999999999

    for symbol in patterns:
        current_distance = get_distance(vector, patterns[symbol])
        if current_distance < best_distance:
            best_distance = current_distance
            best_symbol = symbol

    if best_symbol not in result:
        result[best_symbol] = 0
    result[best_symbol] += 1

    plt.cla()
    plt.title("Class - '" + best_symbol + "'")
    plt.imshow(region.image, cmap='gray')
    plt.savefig(out_path / ('image_' + str(region.label) + '.png'))

print('Распознавание через вектор признаков')
print(result)
print(f'Всего найдено {len(regions)} символов')