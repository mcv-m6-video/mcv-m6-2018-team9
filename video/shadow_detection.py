from skimage.filters import gaussian
from skimage import feature
from skimage.filters import scharr
import numpy as np

'''
Methods for computing the hard shadow mask according to:
L. Xu, F. Qi and R. Jiang, "Shadow Removal from a Single Image,
" Sixth International Conference on Intelligent Systems Design and Applications, 
Jinan, 2006, pp. 1049-1054.
http://lxu.me/mypapers/XuL_ShadowRemoval.pdf
'''


def to_gray_by_max(im):
    '''Returns the value image described by 
    the maximum value from each channel
    in: 
    Image in RGB
    out:
    Value image
    '''
    
    a = im[:,:,0]
    b = im[:,:,1]
    c = im[:,:,2]
    
    m = np.maximum(a,b)
    m = np.maximum(m,c)
    return m

def logImg(im):
    '''
    Computes the logarithmic image
    in: 
    image in RGB
    out: 
    image in RGB (logarithmic values)
    '''
    
    a = im[:,:,0]
    b = im[:,:,1]
    c = im[:,:,2]
    
    on = np.ones(a.shape)

    a = a+on
    b = b+on
    c = c+on
    
    a = np.log(a)
    b = np.log(b)
    c = np.log(c)

    a = (a-a.min())/a.max()
    b = (b-b.min())/b.max()
    c = (c-c.min())/c.max()

    x = np.dstack([a,b,c])
    return x

def finlayson(a,b,c, alpha):
    '''
    Computes the finlayson image (acording to the paper)
    in: 
    a: red channel
    b: blue channel
    c: green channel
    alpha: projection angle
    out:
    grayscale image
    '''
    mask = np.exp(np.cos(alpha)*np.log(a/b) + np.sin(alpha)*np.log(c/b))
    return mask

def HS(im_e, m1_e, m2_e, t1, t2):
    '''
    Computes the hard shadow mask
    in:
    im_e: edges of the input image
    m1_e: edges of invariant image 1
    m2_e: edges of invariant image 2
    t1: tolerance for the input image
    t2: tolerance for the invariant images
    out:
    binary image corresponding to the shadow mask
    '''
    mask = np.zeros(im_e.shape, dtype='uint8')
    
    i1 = np.where(np.minimum(m1_e,m2_e) < t2 )
    mask[i1] = 1
    i2 = np.where(im_e <= t1)
    mask[i2] = 0
    return mask


def prepare_channels(im):
    '''
    Add one to each channel in order to perform logarithms and divisions
    in:
    rgb image
    out:
    translated rgb image
    '''
    
    a = im[:,:,0]
    b = im[:,:,1]
    c = im[:,:,2]
    
    on = np.ones(a.shape)

    a = a+on
    b = b+on
    c = c+on
    
    return (a,b,c)
    
def color_norm(a,b,c):
    '''
    Perform a rgb normalization
    in: 
    a: translated r channel
    b: translated blue channel
    c: translated green channel
    out:
    normalized rgb image
    '''
    
    norm = 1./np.sqrt(np.square(a) + np.square(b) + np.square(c))

    a = a*norm
    b = b*norm
    c = c*norm

    return np.dstack([a,b,c])   

def get_shadow_mask(im, t1, t2, alpha = 2.5, sigma_1 = 5, sigma_2 = 0.4):
    '''
    Obtains the shadow mask for one image
    in:
    im: rgb image
    t1: t1 for HS
    t2: t2 for HS
    alpha: alpha for finlayson
    sigma_1: sigma of the gaussian filter to the input image
    sigma_2: sigma for the gaussian filter to the normalized image
    out:
    binary mask
    '''

    im = logImg(im)
    a, b, c = prepare_channels(im)
    
    im_ = gaussian(im, sigma=sigma_1)

    im_max = np.amax(im_, axis = 2)
    
    mask_norm_color = gaussian(np.amax(color_norm(a,b,c), axis = 2), sigma=sigma_2)
    
    mask_finlayson = finlayson(a,b,c, alpha)
    
    im_e = scharr(im_max)
    m1_e = scharr(mask_norm_color)
    m2_e = scharr(mask_finlayson)
    
    return HS(im_e, m1_e, m2_e, t1, t2).astype(bool)

def shadow_batch(batch,t1, t2, alpha = 2.5, sigma_1 = 2, sigma_2 = 0.4):
    '''
    Obtains a shhadow mask for each rgb image in an array
    in:
    im: im for get_shadow_mask
    t1: t1 for get_shadow_mask
    t2: t2 for get_shadow_mask
    alapha: alpha for get_shadow_mask
    sigma_1: sigma_1 for get_shadow_mask
    sigma_2: sigma_2 for get_shadow_mask
    out:
    batch of binary masks
    '''    
    
    return np.array([np.invert(get_shadow_mask(im, t1, t2, alpha = alpha, sigma_1 = sigma_1, sigma_2 = sigma_2))
                             for im in batch])
    