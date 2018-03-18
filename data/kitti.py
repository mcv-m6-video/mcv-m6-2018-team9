import os
import cv2
import numpy as np


def read_gt_file(path):
    """Read an optical flow map from disk

    Optical flow maps are stored in disk as 3-channel uint16 PNG images,
    following the method described in the KITTI optical flow dataset 2012
    (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow).

    Returns:
      numpy array with shape [height, width, 3]. The first and second channels
      denote the corresponding optical flow 2D vector (u, v). The third channel
      is a mask denoting if an optical flow 2D vector exists for that pixel.

      Vector components u and v values range [-512..512].

    NOTE: this function is copied from video/optical_flow.py. We do not move it
    here to keep backwards-compatibility with week 1.

    """
    data = cv2.imread(path, -1).astype('float32')
    result = np.empty(data.shape, dtype='float32')
    result[:,:,0] = (data[:,:,2] - 2**15) / 64
    result[:,:,1] = (data[:,:,1] - 2**15) / 64
    result[:,:,2] = data[:,:,0]

    return result


def read_sequence(number, annotated=True):
    """Read a train sequence from the KITTI dataset

    The sequence belongs to the `image_0` folder, which is a subdirectory of
    the `training` folder. For the ground truth, we use the `flow_noc`
    directory, as stated in the exercises for Week 1 (pre-slides page 33).

    Args:
      number: (int) it identifies the sequence to read. For example, for
        number=3, the sequence is composed by the files 000003_10.png and
        000003_11.png.
      annotated: (bool) when True, the function returns the ground truth for
        that sequence.

    Returns:
      A numpy array with shape [2, h, w] with the 2 images (grayscale) of the
      sequence.

      Additionally, when annotated=True, a second numpy array with shape
      [h, w, 3] is returned with the optical flow ground truth annotation.

    """
    # Kitti directories we use
    kitti_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets',
                             'data_stereo_flow')
    train_dir = os.path.join(kitti_dir, 'training', 'image_0')
    gt_dir = os.path.join(kitti_dir, 'training', 'flow_noc')

    # Paths to images to load
    im0_path = os.path.join(train_dir, f'{number:06d}_10.png')
    im1_path = os.path.join(train_dir, f'{number:06d}_11.png')
    gt_path = os.path.join(gt_dir, f'{number:06d}_10.png')

    # Read and convert images
    im0 = cv2.imread(im0_path, -1)
    im1 = cv2.imread(im1_path, -1)
    seq = np.array([im0, im1])
    seq = seq[..., np.newaxis]  # add extra axis to have same shape as cdnet:
                                # [n, h, w, 1]

    if annotated:
        gt = read_gt_file(gt_path)
        result = (seq, gt)
    else:
        result = seq

    return result
