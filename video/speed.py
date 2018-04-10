import numpy as np
import cv2

    
def speed(sp={}, filters = None, meter_pix=9.14/70.55 , fps=25, matrix = np.eye(3), out_image = None, n_frames = 0):

def speedv3(sp={}, filters = None, meter_pix = 27.43/42.55  , fps=12, matrix = np.eye(3), out_image = None, 
            n_frames = 0, skip_frames = 5):

    """
    params:
      sp: dictionary where the speeds will be stored
      filters: list of kalman filters obtained from the kalman tracker
      meter_pix: scalar conversion. Directly measured from the image with a known distance
      fps: scalar conversor
      matrix: Homography matrix
      out_image: Im supplied, speed will be written in the centroid position
      n_frames: if supplied, frames previous to the object appearence will have 0 speed
      skip_frames: number of frames that are skipped between distance computation
    """        
    
    for kfilt in filters:
        # speed estimator
        try:

            
            sp['cout_skip'+str(kfilt['id'])] += 1
            sp['cout_skip'+str(kfilt['id'])] %= skip_frames
            
            #add centroid and compute distance
            if(sp['cout_skip'+str(kfilt['id'])] == 0):
                
                trans_motion = np.dot(matrix, np.append(kfilt['centroid']-sp[str(kfilt['id'])],np.zeros(1)) )
                dist = np.linalg.norm(trans_motion)

                #pix/frame-> m/s
                speed= dist/skip_frames*meter_pix*fps*3.6
                sp['speed'+str(kfilt['id'])].append(speed)
                sp[str(kfilt['id'])] = kfilt['centroid']

                new_c = np.dot(matrix, np.append(kfilt['centroid'],np.ones(1)))
                new_c/=new_c[2]

                sp['c'+str(kfilt['id'])].append(new_c)                

            if out_image is not None:
        
                cv2.putText(out_image, str(speed), (kfilt['centroid'][0],kfilt['centroid'][1])
                       , cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255)

        except KeyError:

            sp[str(kfilt['id'])] = kfilt['centroid']
            sp['speed'+str(kfilt['id'])] = n_frames*[0]
            sp['c'+str(kfilt['id'])] = n_frames*[np.zeros(2)]
            sp['cout_skip'+str(kfilt['id'])] = 0
            
    if out_image is not None: 
        return out_image
    else:
        return 
        
