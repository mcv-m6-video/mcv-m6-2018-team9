import os
import glob

import cv2
import numpy as np
from PIL import Image


def iter_dataset(dataset, start, end, process_im=None, process_gt=None,
                 bg_th=50, fg_th=85):

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

            img_gt = np.where(gt >= fg_th, 1, 0)
            img_valid_gt = np.where((gt >= fg_th) & (gt <= bg_th), 1, 0)

            gt = np.stack((img_gt,img_valid_gt))

            if process_im:
                im = process_im(im)

            if process_gt:
                gt = process_gt(gt)

            yield (im, gt)


def read_dataset(dataset, start=0, end=-1, colorspace='rgb', annotated=True,
                 bg_th=50, fg_th=85):
    """

    colorspace: (str) 'rgb', 'gray'

    annotated: (bool) if True, returns the batch of images, their
      corresponding ground truth annotations and the ground truth valid
      pixels according to gt labels
      .

    """
    if colorspace == 'rgb':
        process_im = lambda im: im
    elif colorspace == 'gray':
        process_im = lambda im: np.expand_dims(
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), axis=2)
    else:
        raise Exception('Unknown colorspace')

    batch_im = []
    batch_gt = []
    for im, gt in iter_dataset(dataset, start, end, process_im, None, bg_th, fg_th):
        batch_im.append(im)
        batch_gt.append(gt)

    if annotated:
        return (np.array(batch_im), np.array(batch_gt))
    else:
        return np.array(batch_im)
