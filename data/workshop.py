import glob
import os.path

import cv2
import numpy as np


def print_video_summary(path):
    """Print some properties of a video file

    Properties: resolution, fps, total nº of frames, total time.

    Args:
      path: (str) path to the video file. The function can read all the video
        formats supported by opencv (eg. avi, mp4, etc).

    """
    vid = cv2.VideoCapture()
    vid.open(path)
    success, frame = vid.read()

    if success:
        print('resolution:', frame.shape)
        print('fps:', vid.get(cv2.CAP_PROP_FPS))
        print('frames:', vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print('duration: {:.2f} seg'.format((vid.get(cv2.CAP_PROP_FRAME_COUNT)
                                             / vid.get(cv2.CAP_PROP_FPS))))
    else:
        print('Error opening video file:', path)


def iter_video(path, step=1, scale=1):
    """Iterate over the frames of a video file

    Args:
      path: (str) path to the video file. The function can read all the video
        formats supported by opencv (eg. avi, mp4, etc).

      step: (int) select the frames to iterate over. When step > 1, then only
        frames multiple of 'step' are considered.

      scale: (float) scaling factor to apply

    Yields:
      The selected frames as numpy arrays in BGR format.

    """
    vid = cv2.VideoCapture()
    vid.open(path)

    fps = vid.get(cv2.CAP_PROP_FPS)
    freq = 1 / fps
    count = 0

    while True:
        time = freq * count
        success = vid.grab()
        return_frame = (count % step == 0)
        count += 1

        if success:
            if return_frame:
                _, frame = vid.retrieve()

                if scale != 1:
                    frame = cv2.resize(frame, None, None, scale, scale)

                yield (time, frame)
        else:
            break


def write_sequence(src, dst, step=1, scale=1):
    """Extract frames from a video file and write them as images in disk

    Args:
      src: (str) path to the video file. The function can read all the video
        formats supported by opencv (eg. avi, mp4, etc).
      dst: (str) directory path where to save the images. Path is created when
        not exist.
      step: (int) select the frames to iterate over. When step > 1, then only
        frames multiple of 'step' are considered.
      scale: (float) scaling factor to apply.

    """
    os.makedirs(dst, exist_ok=True)
    i = 0
    for ts, frame in iter_video(src, step=step, scale=scale):
        impath = os.path.join(dst, f"frame{i:05d}_ts{ts*1000:06.0f}.png")
        cv2.imwrite(impath, frame)
        print(impath)
        i += step


def read_sequence(name, colorspace='rgb'):
    """Read a workshop sequence and load it into a numpy array

    Args:
      name: (str) either 'sequence1', 'sequence2' or 'sequence3'.
      colorspace: (str) either 'gray' or 'rgb'.

    Returns:
      Numpy array with shape [n_images, h, w, n_channels].
      When colorspace='gray' n_channels is 1.

    """
    root_folder = os.path.join(os.path.dirname(__file__), '..')
    seq_folder = os.path.join(root_folder, 'datasets', 'workshop', name)
    pattern = os.path.join(seq_folder, '*.png')

    im_list = []
    for path in sorted(glob.glob(pattern)):
        im = cv2.imread(path)  # imread returns image in BGR format

        if colorspace == 'rgb':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif colorspace == 'gray':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im[..., np.newaxis]
        else:
            raise ValueError('Unknown value for colorspace')

        im_list.append(im)

    return np.array(im_list, dtype='float32')
