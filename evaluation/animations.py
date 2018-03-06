import numpy as np
import imageio
import cv2 as cv


def video_recorder(images, out_path, out_filename, codec=None, out_ext='gif'):

    """ims -> image batch, numpy array [num_ims, height, width, channel]
       out_path -> path to store recorded video
       out_filename -> name to store the recorded video
       codec -> codec employed to record video. Must agree with out_path
                tested values for video ['mp4v']
        out_ext -> name to store the recorded video
                        tested values for video ['mp4']

        if input images are non greyscale and extension is set to 'gif' the
        animation will be created using only the first channel of the images

    """
    is_mask = False
    if np.uint8(images[...,:]).max() <= 1:
        is_mask = True

    color = False
    if len(images.shape) > 3:
        n, h, w, _ = images.shape
        if images.shape[3] > 1:
            color = True
    else:
        n, h, w = images.shape

    if out_ext == 'gif':
        if len(images.shape) > 3:
            images = images[:,:,:,0]
        if is_mask:
            images = images * 255

        images = np.uint8(images)

        if images.shape[0] > 200:
            with imageio.get_writer(out_path + out_filename + '.' + out_ext,
                                    mode='I') as writer:
                for i,im in enumerate(images):
                    if i < 200:
                        writer.append_data(im)
        else:
            imageio.mimsave(out_path + out_filename + '.' + out_ext, images)


    else:
        fourcc = cv.VideoWriter_fourcc(*codec)
        video_out = cv.VideoWriter(out_path+out_filename+'.'+out_ext, fourcc, 10,
                                   (int(w), int(h)), color)
        print('images shape: '+str(images.shape))
        for i, im in enumerate(images):
            if color:
                print('im shape ',im.shape)
                im_out = np.uint8(im)
            else:
                im_out = np.uint8(im * 255)
            video_out.write(im_out)

        video_out.release()
        cv.destroyAllWindows()