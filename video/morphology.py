import numpy as np
import skimage.morphology as skmorph


def erosion(batch, selem):
    """Apply erosion over a batch of images

    The erosion is applied to each image independently. Each image is a boolean
    mask obtained in the background subtraction process.

    Args:
      batch: numpy arrary of shape [n_images, height, width] and dtype='bool'.

      selem: structuring element to apply. See [1] for predefined constructors.

    Returns:
      A new batch of images with same shape as `batch` and dtype='bool'

    [1] http://scikit-image.org/docs/dev/api/skimage.morphology.html

    """
    result = np.empty_like(batch, dtype='bool')

    for i, im in enumerate(batch):
        result[i] = skmorph.binary_erosion(batch[i], selem)

    return result
