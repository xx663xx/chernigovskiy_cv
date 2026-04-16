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
    temp = np.zeros((region_image.shape[0] + 2, region_image.shape[1] + 2), dtype = bool)
    temp[1:-1, 1:-1] = region_image
    inv = np.logical_not(temp)
    labeled = label(inv)
    return np.max(labeled) - 1

def get_small_image(region_image):
    small = resize(region_image.astype(float), (30, 30), order = 0, anti_aliasing = False, preserve_range = True)
    return (small > 0.5).astype(float)

def get_feature_vector(region_image):
    small = get_small_image(region_image)
    h, w = region_image.shape
    area = region_image.sum() / region_image.size
    aspect = h / w
    holes = count_holes(region_image)
    row_sums = small.sum(axis = 1) / 30
    col_sums = small.sum(axis = 0) / 30
    vector = [area, aspect, holes]
    vector.extend(row_sums.tolist())
    vector.extend(col_sums.tolist())
    return np.array(vector, dtype=float)

def get_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def get_image_distance(image1, image2):
    return np.sqrt(np.sum((image1 - image2) ** 2))

alphabet_names = ['A', 'B', '8', '0', '1', 'W', 'X', '*', '-', '/', 'P', 'D']
alphabet_image = imread(save_path / 'img' / 'alphabet_ext.png')
alphabet_binary = make_binary(alphabet_image, 100, 'lt')
alphabet_labeled = label(alphabet_binary)
alphabet_regions = [region for region in regionprops(alphabet_labeled) if region.area > 20]
alphabet_regions = sort_regions(alphabet_regions)
patterns = {}
pattern_images = {}

for i in range(len(alphabet_regions)):
    symbol = alphabet_names[i]
    patterns[symbol] = get_feature_vector(alphabet_regions[i].image)
    pattern_images[symbol] = get_small_image(alphabet_regions[i].image)

symbols_image = imread(save_path / 'img' / 'symbols.png')
symbols_binary = make_binary(symbols_image, 10, 'gt')
symbols_labeled = label(symbols_binary)
symbol_regions = []
for region in regionprops(symbols_labeled):
    if region.area > 3:
        symbol_regions.append(region)

result = {}

for region in symbol_regions:
    vector = get_feature_vector(region.image)
    small_image = get_small_image(region.image)
    best_symbol = ''
    best_score = 999999999
    for symbol in patterns:
        d1 = get_distance(vector, patterns[symbol])
        d2 = get_image_distance(small_image, pattern_images[symbol])
        score = d1 + d2
        if score < best_score:
            best_score = score
            best_symbol = symbol

    if best_symbol not in result:
        result[best_symbol] = 0
    result[best_symbol] += 1

print('Частотный словарь символов\n')
for symbol in sorted(result):
    print(symbol, result[symbol])

print(f'\nВсего найдено {len(symbol_regions)} символов',)
