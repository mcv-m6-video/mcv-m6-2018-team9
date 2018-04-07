import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from skimage import img_as_ubyte


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
      im: (numpy array) numpy array image in grayscale or color.
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

    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = np.squeeze(im)  # grayscale image shapes: [h, w, 1] -> [h, w]

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


def patch_correlation(template, im):
    """Returns: (x, y) displacement vector
    """
    # https://docs.opencv.org/3.4.1/df/dfb/group__imgproc__object.html

    # TODO: deal with multiple maxima!! It should choose the most centered...
    result = cv2.matchTemplate(im, template, cv2.TM_SQDIFF)
    displ = np.unravel_index(np.argmin(result), result.shape)
    return (displ[1], displ[0])


def block_matching(im1, im2, block_size=16, max_motion=16):

    search_area = 2 * max_motion + block_size
    block_rows = int(im1.shape[0] / block_size)
    block_cols = int(im1.shape[1] / block_size)

    # Add extra row / column with the remainder pixels, when large enough
    if im1.shape[0] % block_size >= 8:
        block_rows += 1

    if im1.shape[1] % block_size >= 8:
        block_cols += 1

    result = np.zeros((im1.shape[0], im1.shape[1], 3), dtype='int16')
    for i in range(block_rows):
        for j in range(block_cols):
            x1 = j * block_size
            y1 = i * block_size
            xa = x1 - max_motion
            ya = y1 - max_motion
            patch = im1[y1:y1 + block_size, x1:x1 + block_size]
            area = im2[max(ya, 0):(ya + search_area),
                       max(xa, 0):(xa + search_area)]
            x2, y2 = patch_correlation(patch, area)
            motion = (max(xa, 0) + x2 - x1, max(ya, 0) + y2 - y1, 1)

            result[y1 : y1 + block_size, x1 : x1 + block_size] = motion

    return result


def block_matching_sequence(seq, block_size=16, max_motion=16):

    n, h, w, _ = seq.shape
    result = np.empty((n - 1, h, w, 3), dtype='int16')
    for i in range(seq.shape[0] - 1):
        result[i] = block_matching(seq[i], seq[i+1], block_size=block_size,
                                   max_motion=max_motion)

    return result


def lucas_kanade(im1, im2, wsize, track_specs={}):

    track_points = np.mgrid[0:im1.shape[0],
                   0:im1.shape[1]].swapaxes(0, 2).swapaxes(0, 1).reshape(
        (im1.shape[0] * im1.shape[1], 2))

    track_points = np.flip(track_points, axis=1)

    track_points = track_points.astype(np.float32)

    out_points, __, __ = cv2.calcOpticalFlowPyrLK(im1, im2, track_points, None,
                                                  winSize=wsize, maxLevel=0,
                                                  criteria=
                                                  (cv2.TERM_CRITERIA_EPS |
                                                   cv2.TERM_CRITERIA_COUNT,
                                                   10, 0.03))

    track_points = np.flip(track_points, axis=1)
    out_points = np.flip(out_points, axis=1)

    track_points = track_points.astype(int)
    out_points = out_points.astype(int)

    shap = list(im1.shape)
    shap = shap + [3]
    shap = tuple(shap)
    flow = np.zeros(shap)
    for a, b in zip(track_points, out_points):
        bf = np.flip(b, axis=0)
        af = np.flip(a, axis=0)
        flow[a[0], a[1], 0:2] = bf - af
        flow[a[0], a[1], 2] = 1

    return flow


def lk_sequence(seq, wsize, track_specs={}):

    n, h, w, _ = seq.shape
    seq = seq[:, :, :, 0]
    result = np.empty((n, h, w, 3), dtype='int16')
    for i in range(seq.shape[0] - 1):
        result[i] = lucas_kanade(seq[i], seq[i+1], wsize, track_specs)

    return result


def lucas_kanade_pyr(im1, im2, wsize, levels, track_specs={}):
    track_points = np.mgrid[0:im1.shape[0],
                   0:im1.shape[1]].swapaxes(0, 2).swapaxes(0, 1).reshape(
        (im1.shape[0] * im1.shape[1], 2))

    track_points = np.flip(track_points, axis=1)

    track_points = track_points.astype(np.float32)

    out_points, __, __ = cv2.calcOpticalFlowPyrLK(im1, im2, track_points, None,
                                                  winSize=wsize,
                                                  maxLevel=levels,
                                                  criteria=
                                                  (cv2.TERM_CRITERIA_EPS |
                                                   cv2.TERM_CRITERIA_COUNT,
                                                   10, 0.03))

    track_points = np.flip(track_points, axis=1)
    out_points = np.flip(out_points, axis=1)

    track_points = track_points.astype(int)
    out_points = out_points.astype(int)

    shap = list(im1.shape)
    shap = shap + [3]
    shap = tuple(shap)
    flow = np.zeros(shap)
    for a, b in zip(track_points, out_points):
        bf = np.flip(b, axis=0)
        af = np.flip(a, axis=0)
        flow[a[0], a[1], 0:2] = bf - af

    return flow


def lk_pyr_sequence(seq,  wsize, pyr_levels, track_specs={}):

    n, h, w, _ = seq.shape
    seq = seq[:, :, :, 0]
    result = np.empty((n, h, w, 3), dtype='int16')
    for i in range(seq.shape[0] - 1):
        result[i] = lucas_kanade_pyr(seq[i], seq[i+1], wsize, pyr_levels,
                                     track_specs)

    return result


def farneback(im1, im2, levels, pyr_sc, wsize, n_iter, poly_n, p_sigma, flag):
    flow = None

    flow = cv2.calcOpticalFlowFarneback(im1, im2, flow, pyr_scale=pyr_sc,
                                        levels=levels, winsize=wsize,
                                        iterations=n_iter, poly_n=poly_n,
                                        poly_sigma=p_sigma,
                                        flags=flag)
    return flow


def farneback_sequence(seq, levels, pyr_sc, wsize, n_iter, poly_n, p_sigma,
                       gauss):
    flag = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if gauss \
        else cv2.OPTFLOW_USE_INITIAL_FLOW

    n, h, w, _ = seq.shape
    seq = seq[:, :, :, 0]
    result = np.empty((n, h, w, 2), dtype='int16')
    for i in range(seq.shape[0] - 1):
        result[i] = farneback(seq[i], seq[i + 1], levels, pyr_sc, wsize, n_iter,
                              poly_n, p_sigma, flag)

    return result



def stabilize(ims_, gt = None, mode = "bac"):
    ''''
    Stabilizes a batch of images using Optical flow (block matching) 
    using block matching. Returns stabilized images with stabilized 
    gt and masks of valid entries
    inputs:
    ims_: (ndarray) Batch of images
    gt: (ndarray) Ground truth 
    mode: (str) bac or forward
    outputs:
    stabilized images
    valid pixels masks
    stabilized gt
    ''' 
    #fix first image
    #for each image, match with previous one
    
    #Act on the gt as well!
    #Use numpy concat?
    
    stabilized_images= np.zeros(ims_.shape).astype('uint8')
    masks = np.zeros(ims_.shape[:3], dtype=bool)
    
    n_im = ims_.shape[0]
    
    if(mode == "bac"):
        ind = ims_.shape[0]
        sign = 1
    else:
        ind = 1
        sign = -1
        
    stabilized_images[n_im-ind] = ims_[n_im-ind]
    current_image = ims_[n_im-ind]
    
    true_mask = np.full((ims_.shape[1:3]), True)
    masks[n_im-ind] = true_mask

    if type(None) != type(gt):
        
        stabilized_gt = np.zeros(gt.shape, dtype = bool)
        stabilized_gt[n_im-ind] = gt[n_im-ind]

        bmsk = gt[n_im-ind][0]
        fgmsk = gt[n_im-ind][1]

    for i in range(1, ims_.shape[0]):
        
        of = block_matching(current_image, ims_[n_im-ind + sign*i])
        u,v = of[:,:,0] , of[:,:,1]
        
        tform = SimilarityTransform(translation=(-u.mean() , -v.mean() ))
        current_image = (255*warp(ims_[n_im-ind + sign*i], tform.inverse)).astype('uint8')
        
        current_mask = warp(true_mask, tform.inverse).astype('bool')

        if type(None) != type(gt):
            
            bmsk = warp(gt[n_im-ind + sign*i][0], tform.inverse).astype('bool')
            fgmsk = warp(gt[n_im-ind + sign*i][1], tform.inverse).astype('bool')
            
            #stabilized_gt[n_im-ind + sign*i] = warp(current_gt, tform.inverse).astype('bool')
            stabilized_gt[n_im-ind + sign*i][0] = bmsk
            stabilized_gt[n_im-ind + sign*i][1] = fgmsk
        
        stabilized_images[n_im-ind + sign*i] = current_image
        masks[n_im-ind + sign*i] = current_mask
        #current_gt = stabilized_gt[n_im-ind + sign*i]
        
    if type(None) != type(gt):    
        
        return stabilized_images, masks, stabilized_gt
    
    return stabilized_images, masks
