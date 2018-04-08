import numpy as np
from skimage.transform import warp


#Define p corr to an image of max distance between pairs (in x and y)
#Correspondences Hand picked!
def p(p):
    return np.array([p[0], p[1], 1])

def DLT(ims, coords = [], dataset = None):
    
    
    '''
    Homography for flattening straight roads.
    Params= 
            ims: list of images
            coords: 
                list of tuples: 1st coordinate of future top left vertex
                                2nd coordinate of future top right vertex
                                3rd coordinate of future bottom left vertex
                                3rd coordinate of future bottom right vertex
            dataset: optional, highway or traffic (only highway tested! 
            do not supply coords if this selected)
    '''
    if len(coords) == 4:
        
        p_11 = p(coords[0])
        p_12 = p(coords[1])
        p_13 = p(coords[2])
        p_14 = p(coords[3])
        
    elif dataset == 'highway':

        #narrow
        p_11 = p((127.304,83.221))
        p_12 = p((261.282, 98.1241))

        #bigger
        p_11 = p((197.512, 19.221))
        p_12 = p((269.282, 22.1241))
        p_13 = p((24.225, 174.857))
        p_14 = p((255.404, 194.516))

    #not implemented    
    elif dataset == 'traffic':
        
        #narrow
        p_11 = p((127.304,83.221))
        p_12 = p((261.282, 98.1241))

        #bigger
        p_11 = p((197.512, 19.221))
        p_12 = p((269.282, 22.1241))
        p_13 = p((24.225, 174.857))
        p_14 = p((255.404, 194.516))
        
    else:
        print("couldn't definie coordinates!")
        return
    
    p1 = np.array([p_11, p_12, p_13, p_14])

    p_21 =  p((0,0))
    p_22 = p((max(p1[:,0]) - min(p1[:,0]), 0))
    p_23 = p((0, max(p1[:,1]) - min(p1[:,1])))
    p_24 = p((max(p1[:,0]) - min(p1[:,0]), max(p1[:,1]) - min(p1[:,1])))    
  

    p2 = np.array([p_21, p_22, p_23, p_24])
    
    
    #DLT
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    inm = np.linalg.inv(H)
    
    output_shape = np.dot(inm,p_14)
    output_shape/= output_shape[2]
    
    if len(ims.shape) == 4:
        wims = np.zeros((ims.shape[0], output_shape[1].astype(int), output_shape[0].astype(int),ims.shape[3]) )
    else:
        wims = np.zeros((ims.shape[0], output_shape[1].astype(int), output_shape[0].astype(int)) )
    
    for i in range(ims.shape[0]):
        wims[i] = warp(ims[i], inm, output_shape=output_shape[:2][::-1].astype(int))
        
    return wims

        


