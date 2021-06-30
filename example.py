from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from skimage.segmentation import mark_boundaries
from pysnic.algorithms.snic import snic
from asnic import generate_seeds, mark_seed_points

# load image
rgb_image = np.array(Image.open("images/G-MYUP - 1995 Letov LK-2M Sluka.jpg"))

# convert the image from RGB to CIELAB
lab_image = skimage.color.rgb2lab(rgb_image)

# generate seeds (lab_image should be numpy.ndarray)
seeds = generate_seeds(lab_image)

# show seed placement
# This will block the execution until you close the figure.
image_of_seed_placement = mark_seed_points(rgb_image, seeds)
plt.figure("ASNIC seed placement with %d seeds" % len(seeds))
plt.imshow(image_of_seed_placement)
plt.savefig("centroids", dpi=300)
plt.show()

# SNIC parameters
compactness = 10.00

# SNIC
# lab_image should be a list for 'snic'
lab_image = lab_image.tolist()
number_of_pixels = rgb_image.shape[0] * rgb_image.shape[1]

segmentation, _, centroids = snic(
    lab_image, seeds, compactness,
    update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))
actual_number_of_segments = len(centroids)

# show the output of SNIC
fig = plt.figure("ASNIC with %d segments" % actual_number_of_segments)
plt.imshow(mark_boundaries(rgb_image, np.array(segmentation)))
plt.savefig("ASNIC with %d segments" % actual_number_of_segments, dpi=300)
plt.show()

