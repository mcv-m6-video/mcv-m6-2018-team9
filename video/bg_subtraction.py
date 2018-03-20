import numpy as np
import cv2 as cv


def create_model(images):
    """Compute the model parameters from a given batch of images

    Args:
      images: numpy array with shape [num_ims, height, width, channel]

    Returns:
      A tuple (mean, std) with the parameters of the new model. Both elements
      are numpy arrays with same shape as images and dtype='float32'.

    """
    mean = np.mean(images, axis=0, dtype='float32')
    std = np.std(images, axis=0, dtype='float32')
    return (mean, std)

def create_model_mask(images, masks):
    '''
    Creates the model taking into account only pixels defined in 
    masks array
    '''
    #We create the masks for masking with numpy
    channeled_masks = np.repeat(masks, 3)
    
    #True values in the mask are masked, so we invert the GT
    masked = ma.masked_array(images.flatten(), mask=np.invert(channeled_masks.astype(bool))).reshape(images.shape)
    mean = np.mean(masked, axis=0, dtype='float32')
    std = np.std(masked, axis=0, dtype='float32')

    return (mean.compressed().reshape(images.shape[1:]), std.compressed().reshape(images.shape[1:]))

def predict(images, model, alpha, rho=0, return_model=False):
    """Apply background subtraction to a batch of images

    Args:
      images: numpy array with shape [num_ims, height, width, channel]
        containing the images to process.
      model: tuple (mean, std) obtaining with create_model() function.
      alpha: (float) threshold which controls when a pixel is foreground
      rho: (float) adaptive factor used to modify model as the batch of images
        is analized.
      return_model: (bool) if True, the final model is returned in addition to
        the predictions.

    Returns:
      A numpy array with shape [num_ims, height, width] and dtype 'bool', with
      the background prediction. True denotes the corresponding pixel is
      background and False denotes it is foreground.

      When return_model=True, a second element is returned with the model
      parameters (mean, std) after analyze the images. Useful to analyze the
      model adaptation when rho != 0.

    """
    n, h, w, _ = images.shape
    mean = model[0].copy()
    std = model[1].copy()
    estimation = np.zeros((n, h, w), dtype='bool')

    for i, im in enumerate(images):
        fgmask = np.absolute(im - mean) >= alpha * (std + 2)
        estimation[i] = np.prod(fgmask, axis=-1, dtype='bool')

        if rho != 0:
            mean[~fgmask] = rho * im[~fgmask] + (1 - rho) * mean[~fgmask]
            std[~fgmask] = np.sqrt(rho * (im[~fgmask] - mean[~fgmask])**2 +
                                   (1 - rho) * std[~fgmask]**2)

    if return_model:
        return (estimation, (mean, std))
    else:
        return estimation
    
def predict_masked(images, masks, model, alpha, rho=0, return_model=False):
    """Apply background subtraction to a batch of images
        Taking into account valid pixels defined in masks array
    Args:
      images: numpy array with shape [num_ims, height, width, channel]
        containing the images to process.
      masks: boolean numpy array defining position of valid pixels
      model: tuple (mean, std) obtaining with create_model() function.
      alpha: (float) threshold which controls when a pixel is foreground
      rho: (float) adaptive factor used to modify model as the batch of images
        is analized.
      return_model: (bool) if True, the final model is returned in addition to
        the predictions.

    Returns:
      A numpy array with shape [num_ims, height, width] and dtype 'bool', with
      the background prediction. True denotes the corresponding pixel is
      background and False denotes it is foreground.

      When return_model=True, a second element is returned with the model
      parameters (mean, std) after analyze the images. Useful to analyze the
      model adaptation when rho != 0.

    """    
    n, h, w, _ = images.shape
    mean = model[0].copy()
    std = model[1].copy()
    estimation = np.zeros((n, h, w), dtype='bool')

    channeled_masks = np.repeat(masks, 3)

    for i, im in enumerate(images):
        fgmask = np.absolute(im - mean) >= alpha * (std + 2)
        estimation[i] = np.logical_and(np.prod(fgmask, axis=-1, dtype='bool'), channeled_masks[i])
        inverted =  np.invert(channeled_masks[i])
        fgmask = np.logical_or(fgmask, inverted)
        if rho != 0:
            mean[~fgmask] = rho * im[~fgmask] + (1 - rho) * mean[~fgmask]
            std[~fgmask] = np.sqrt(rho * (im[~fgmask] - mean[~fgmask])**2 +
                                   (1 - rho) * std[~fgmask]**2)
    
    if return_model:
        return (estimation, (mean, std))
    else:
        return estimation
    
    

def create_model_opencv(threshold):
    return cv.bgsegm.createBackgroundSubtractorLSBP(LSBPthreshold=threshold)


def predict_opencv(images, model):
    mask_batch = []
    for image in images:
        mask = model.apply(image)
        mask_batch.append(mask)
    return np.array(mask_batch)
