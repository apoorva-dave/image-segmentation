import numpy as np
from skimage import data,io
coins = data.coins()
histo = np.histogram(coins, bins=np.arange(0, 256))
from skimage.feature import canny
edges = canny(coins/255.)
from scipy import ndimage as ndi
fill_coins = ndi.binary_fill_holes(edges)
label_objects, nb_labels = ndi.label(fill_coins)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]
io.imshow(coins_cleaned)
io.show()