import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_file(path):
    """Read an optical flow map from disk

    Optical flow maps are stored in disk as 3-channel uint16 PNG images,
    following the method described in the KITTI optical flow dataset 2012
    (http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow).

    Returns:
      numpy array with shape [height, width, 3]. The first and second channels
      denote the corresponding optical flow 2D vector (u, v). The third channel
      is a mask denoting if an optical flow 2D vector exists for that pixel.

      Vector components u and v values range [-512..512].

    """
    data = cv2.imread(path, -1).astype('float32')
    result = np.empty(data.shape, dtype='float32')
    result[:,:,0] = (data[:,:,2] - 2**15) / 64
    result[:,:,1] = (data[:,:,1] - 2**15) / 64
    result[:,:,2] = data[:,:,0]

    return result


def quantize_map(flow, size):
    """Quantize an optical flow map

    The map is divided into non-overlapping square areas and the mean motion
    vector is computed for each of them.

    Args:
      flow: numpy array with the optical flow to quantize
      size: (int) size of the square areas to use in the process.

    Returns:
      Numpy array with shape [new_h, new_w, 3], where:
        new_h = int(flow.shape[0] / size)
        new_w = int(flow.shape[1] / size)

    """
    h, w, n = flow.shape

    h_dst = int(h / size)
    w_dst = int(w / size)
    dst = np.zeros([h_dst, w_dst, n], dtype='float32')

    for i in range(h_dst):
        for j in range(w_dst):
            bin = flow[i*size:(i+1)*size, j*size:(j+1)*size]
            valid = bin[:,:,2] == 1

            if bin[valid].size > 0:
                dst[i, j] = np.mean(bin[valid], axis=0)

    return dst


def plot_map(im, flow, size=None, title=''):
    """Plot an optical flow map on top of their corresponding image

    Args:
      im: (numpy array) image in grayscale or color
      flow: (numpy array) optical flow map for `im`
      size: (optional, int) the size to use in the quantization process. When
        specified, the image is divided into non-overlapping square areas, and
        the mean optical motion vector is computed for each.
      title: (optional, str) plot title.

    """
    if size:
        flow = quantize_map(flow, size)
        start = int(size/2)
    else:
        start = 0
        size = 1

    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    h, w = flow.shape[:2]
    x, y = np.meshgrid(np.arange(w) * size + start,
                       np.arange(h) * size + start)
    valid = flow[:,:,2] == 1

    fig = plt.figure()
    plt.title(title)
    plt.imshow(im, cmap='gray')
    plt.quiver(x[valid], y[valid], flow[valid][:,0], -flow[valid][:,1],
               angles='uv', minlength=0.5, width=0.002, headwidth=4,
               color='#ff6600ff')
    plt.show()
    return fig
