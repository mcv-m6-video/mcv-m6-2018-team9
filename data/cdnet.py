import os
import glob

import cv2
import numpy as np
from PIL import Image


LABEL_STATIC = 0
LABEL_SHADOW = 50
LABEL_OUTSIDE_ROI = 85
LABEL_UNKWN_MOTION = 170
LABEL_MOTION = 255


def iter_dataset(dataset, start, end, process_im=None, process_gt=None,
                 bg_th=50, fg_th=255):

    if dataset not in ['fall', 'highway', 'traffic']:
        raise Exception('Unknown dataset')

    root_folder = os.path.join(os.path.dirname(__file__), '..')
    im_folder = os.path.join(root_folder, 'datasets', dataset, 'input')
    gt_folder = os.path.join(root_folder, 'datasets', dataset, 'groundtruth')

    for i in range(start, end):
        im_path = os.path.join(im_folder, 'in{:06d}.jpg'.format(i))
        gt_path = os.path.join(gt_folder, 'gt{:06d}.png'.format(i))

        if os.path.exists(im_path) and os.path.exists(gt_path):
            im = cv2.imread(im_path)  # imread returns image in BGR format
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            gt = Image.open(gt_path)
            gt = np.array(gt)
            gt = np.stack([(gt >= fg_th),
                           (gt >= fg_th) | (gt <= bg_th)])

            if process_im:
                im = process_im(im)

            if process_gt:
                gt = process_gt(gt)

            yield (im, gt)


def read_dataset(dataset, start=0, end=-1, colorspace='rgb', annotated=True,
                 bg_th=50, fg_th=255):
    """

    colorspace: (str) 'rgb', 'gray'

    annotated: (bool) if True, returns the batch of images, their
       corresponding ground truth annotations and the ground truth valid
       pixels according to gt labels

    """
    if colorspace == 'rgb':
        process_im = lambda im: im
    elif colorspace == 'ycbcr':
        process_im = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
    elif colorspace == 'ycbcr-only-color':
        process_im = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)[:, :, 1:]
    elif colorspace == 'hsv':
        process_im = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    elif colorspace == 'hsv-only-color':
        process_im = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2HSV)[:, :, :2]
    elif colorspace == 'gray':
        process_im = lambda im: np.expand_dims(
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), axis=2)
    else:
        raise Exception('Unknown colorspace')

    batch_im = []
    batch_gt = []
    for im, gt in iter_dataset(dataset, start, end, process_im=process_im,
                               bg_th=bg_th, fg_th=fg_th):
        batch_im.append(im)
        batch_gt.append(gt)

    if annotated:
        return (np.array(batch_im), np.array(batch_gt))
    else:
        return np.array(batch_im)


def read_sequence(week, dataset, split, colorspace='rgb', annotated=True,
                  bg_th=50, fg_th=255):
    """Read the sequence of images to use for a given dataset

    Given a CDNET dataset, the sequence of images to read is determined by
    `week` and `split` parameters.

    Args:
      week: (str) either 'week1', 'week2' or 'week3'.
      dataset: (str) either 'highway', 'fall' or 'traffic'.
      split: (str) either 'train' or 'test'.
      colorspace: (str) the colorspace for the returned images. Value: one of
        'rgb', 'ycbr', 'ycbr-only-color', 'hsv', 'hsv-only-color', 'gray'.
      annotated: (bool) when True, the function return the ground truth labels
        in addition to the batch of images.
      bg_th: (int) threshold to set background labels. All pixels with value <=
        bg_th are labeled as background. Only used when annotated=True.
      fg_th: (int) threshold to set foreground labels. All pixels with value >=
        fg_th are labeled as foreground. Only used when annotated=True.

    Returns:
      A numpy array with shape [n_images, h, w, n_channels] with the images for
      the selected sequence.

      Additionally, when annotated=True, a second numpy array is returned, with
      shape [n_images, h, w] and dtype='bool', containing the ground truth
      annotations.

    """

    if week in ['week1', 'week2', 'week3','week4']:
        dataset_train_idx = {'highway': (1050, 1200),
                             'fall': (1460, 1510),
                             'traffic': (950, 1000)}
        dataset_test_idx = {'highway': (1200, 1350),
                            'fall': (1510, 1560),
                            'traffic': (1000, 1050)}

    elif week == 'week5':
        dataset_train_idx = {'highway': (1050, 1350),
                             'traffic': (950, 1050)}
        # TODO: choose range
        dataset_test_idx = {'highway': (1, 500),  # 1-1570
                            'traffic': (1, 500)}  # 1-1700

    else:
        raise ValueError('Invalid value for week')

    if split == 'train':
        start, end = dataset_train_idx[dataset]
    elif split == 'test':
        start, end = dataset_test_idx[dataset]

    seq = read_dataset(dataset, start=start, end=end, colorspace=colorspace,
                       annotated=annotated, bg_th=bg_th, fg_th=fg_th)
    return seq
