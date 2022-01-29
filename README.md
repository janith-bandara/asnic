# ASNIC: Adaptive Centroid Placement Based SNIC for Superpixel Segmentation

Python implementation of the adaptive seed (centroid) placement part in [ASNIC](https://ieeexplore.ieee.org/document/9185361) algorithm.

`asnic` produce seeds such a way that it captures the features in the image by taking the information distribution of the image into account, in contrast to `snic` algorithm which just place seeds in a regular grid. `snic` also oversegment the uniform regions unnecessarily.
## Example Segmentations with ASNIC and SNIC 

Images in the **Left** are the segmentations produced by **ASNIC** and the images in the **right** are the segmentations produced by **SNIC** with similar number of segments. 
![](https://github.com/janith-bandara/asnic/blob/main/results/Comparison%20Puffin.jpg)
![](https://github.com/janith-bandara/asnic/blob/main/results/Comparison%20Plane.jpg)

Following figure shows the corresponding seeds produced by `asnic` algorithm. It is clear that seeds are placed in adaptive manner which essentially capture the relevant features.
In `snic` algorithm the seeds are placed in a regular grid which does not take the information distribution in the image to account.
![](https://github.com/janith-bandara/asnic/blob/main/results/Seed%20Placement.jpg)
 
See [results](https://github.com/janith-bandara/asnic/tree/main/results) folder for example segmentations.
The images used here are taken from [Flicker](https://www.flickr.com) and all of them are in public domain. References to the used images are given at the bottom.

## Usage

To generate seeds:
```python
from asnic import generate_seeds
import numpy as np
from PIL import Image
import skimage.color

# load image
rgb_image = np.array(Image.open("image.jpg"))
# convert the image from RGB to CIELAB 
lab_image = skimage.color.rgb2lab(rgb_image)
# generate seeds (lab_image should be numpy.ndarray)
seeds = generate_seeds(lab_image) 
```
Additionally, you can pass additional parameters to the method `generate_seeds` as follows:
```python
# generate seeds
seeds = generate_seeds(
    lab_image,
    block_size=(100, 100),
    entropy_tr=8,
    print_status=True
)
```

If you want to see how the seed points are placed on the image, you can use `mark_seed_points` method.
```python
from asnic import mark_seed_points
import matplotlib.pyplot as plt

# show seed placement
# This will block the execution until you close the figure.
image_of_seed_placement = mark_seed_points(rgb_image, seeds)
plt.figure("Seed Placement")
plt.imshow(image_of_seed_placement)
plt.show()
```

The seeds generated by the method `generate_seeds` then can be passed for the `seeds` parameter in the `snic` superpixel segmentaion algorithm.
```python
segmentation, _, centroids = snic(
    lab_image, seeds, compactness,
    update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))
```

See `example.py` for example usage.

To install the Python implementation of the SNIC algorithm (PySNIC) using `pip`:
```python
pip install pysnic
```


For more details on how to use SNIC implementation please refer to https://github.com/MoritzWillig/pysnic

**We have only implemented the seed generation part of the ASNIC algorithm. To perform a complete segmentation, additionally, you should use the implementation of the SNIC algorithm** (https://github.com/MoritzWillig/pysnic).


## Publications
The seed generator implemented here is based on the following publication:

["Adaptive Centroid Placement Based SNIC for Superpixel Segmentation"](https://ieeexplore.ieee.org/document/9185361)
```
@INPROCEEDINGS{9185361,  
  author={Bandara Senanayaka, Janith and Thilanka Morawaliyadda, Dilshan and Tharuka Senarath, Shehan and Indika Godaliyadda, Roshan and Parakrama Ekanayake, Mervyn},  
  booktitle={2020 Moratuwa Engineering Research Conference (MERCon)},   
  title={Adaptive Centroid Placement Based SNIC for Superpixel Segmentation},   
  year={2020},  
  volume={},  
  number={},  
  pages={242-247},  
  doi={10.1109/MERCon50084.2020.9185361}}
```

The SNIC algorithm is based on the following publication:

["Superpixels and Polygons using Simple Non-Iterative Clustering"](https://ieeexplore.ieee.org/document/8100003)
```
@inproceedings{snic_cvpr17,
  author = {Achanta, Radhakrishna and Susstrunk, Sabine},
  title = {Superpixels and Polygons using Simple Non-Iterative Clustering},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}
```


## References to the Images Used Here
All the images used here are from the public domain.

- T. Niedzwiedz. "Plumage Hygiene | An Atlantic puffin on the island of Mykine… | Flickr." 
https://www.flickr.com/photos/tomek_niedzwiedz/29347962448 (accessed Jun. 26, 2021)
  
- R. Mitchell. "G-MYUP | 1995 Letov LK-2M Sluka | Rob Mitchell | Flickr." 
https://www.flickr.com/photos/swallowedtail/32582973873 (accessed Jun. 26, 2021)

## Repository
GitHub repository link: https://github.com/janith-bandara/asnic
