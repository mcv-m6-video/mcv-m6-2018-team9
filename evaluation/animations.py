import numpy as np
import imageio
import cv2 as cv


def video_recorder(images, out_path, out_filename, codec=None, out_ext='gif',
                    is_mask=False):

    """ims -> image batch, numpy array [num_ims, height, width, channel]
       out_path -> path to store recorded video
       out_filename -> name to store the recorded video
       codec -> codec employed to record video. Must agree with out_path
                tested values for video ['mp4v']
        out_ext -> name to store the recorded video
                        tested values for video ['mp4']


       is_mask -> if true multiply values by 255

    """
    if out_ext == 'gif':
        out_im = []
        for im in images:
            if is_mask:
                out_im.append(np.uint8(im * 255))
            else:
                out_im.append(np.uint8(im))

        imageio.mimsave(out_path + out_filename+'.'+out_ext, out_im)

    else:
        color = False
        if len(images.shape) > 3:
            color = True
            n, h, w, _ = images.shape
        else:
            n, h, w = images.shape

        fourcc = cv.VideoWriter_fourcc(*codec)
        video_out = cv.VideoWriter(out_path+out_filename+'.'+out_ext, fourcc, 10,
                                   (int(w), int(h)), color)

        for i, im in enumerate(images):
            if color:
                im_out = np.uint8(im)
            else:
                im_out = np.uint8(im * 255)
                #im_out = np.uint8(cv.flip(im * 255, 1))
            video_out.write(im_out)

        video_out.release()
        cv.destroyAllWindows()