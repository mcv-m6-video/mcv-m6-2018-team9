import numpy as np
import cv2 as cv


def create_model(images):
    """
    images -> image batch, numpy array [n, height, width, channel]

    """
    mean = np.mean(images, axis=0, dtype='float32')
    std = np.std(images, axis=0, dtype='float32')
    return (mean, std)


def predict(images, model, alpha, rho=0):
    """ims -> image batch, numpy array [num_ims, height, width, channel]
    model -> 2-tuple (mean, std) each numpy array [height, width, channel]
    alpha -> scalar

    NOTE: when rho is provided and not equal to 0, the input model is updated
    by the function. If you want to keep the original model, make a copy before
    invoke the function:

    >>> model_orig = model.copy()  # keep original parameters in model_orig
    >>> estimation = predict(images, model, 2, rho=0.5)  # model is modified

    Returns:
      A numpy array with shape [num_ims, height, width] and dtype 'bool', with
      the background prediction. True denotes the corresponding pixel is
      background and False denotes it is foreground.

    """
    n, h, w, _ = images.shape
    mean, std = model
    threshold = alpha * (std + 2)
    estimation = np.zeros((n, h, w), dtype='bool')

    for i, im in enumerate(images):
        estimation[i] = np.prod(np.absolute(im - mean) >= threshold,
                                axis=-1, dtype='bool')
        if rho != 0:
            # use arr[:,:,:] to force side effect (namely: modify input model)
            mean[:,:,:] = rho * im + (1 - rho) * mean
            std[:,:,:] = rho * (im - mean) + (1 - rho) * std

    return estimation


def create_model_opencv():
    return cv.bgsegm.createBackgroundSubtractorLSBP()


def predict_opencv(images, model):
    mask_batch = []
    for image in images:
        mask = model.apply(image)
        mask_batch.append(mask)
    return np.array(mask_batch)