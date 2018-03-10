import cv2
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


def imfill(batch, neighb=4):
    """Fill the holes in a batch of binary images

    Args:
      batch: numpy arrary of shape [n_images, height, width] and dtype='bool'.
      neighb: (int) can be 4 or 8, defines the neighbourhood connectivity which
        determines.

    Returns:
      A numpy array with same shape as batch and dtype='bool', containing the
      filled images.

    """
    result = np.empty_like(batch, dtype='bool')

    for i, im in enumerate(batch):
        __, cc = cv2.connectedComponents(im.astype('uint8'), neighb)
        h, w = im.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype='uint8')
        cv2.floodFill(cc, mask, (0,0), 1, loDiff=0, upDiff=0,
                      flags=cv2.FLOODFILL_MASK_ONLY)
        result[i] = ~(mask[1:-1, 1:-1].astype('bool'))

    return result


def filter_small(batch, min_area, neighb=4):
    result = np.empty_like(batch, dtype='bool')

    for i, im in enumerate(batch):
        n_labels, cc = cv2.connectedComponents(im.astype('uint8'), neighb)
        h, w = im.shape[:2]
        result[i] = np.zeros((h, w), dtype='uint8')

        for lab in range(n_labels):
            if lab == 0:
                continue
            if np.count_nonzero(cc == lab) > min_area:
                result[i][np.where(cc == lab)] = 1

    return result
