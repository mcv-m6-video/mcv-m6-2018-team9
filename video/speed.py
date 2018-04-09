import numpy as np
import cv2


def speed(sp={}, filters = None, meter_pix=9./123.55 , fps=12, matrix = np.eye(3), out_image = None, n_frames = 0):
    """
    params:
      sp: dictionary where the speeds will be stored
      filters: list of kalman filters obtained from the kalman tracker
      meter_pix: scalar conversion. Directly measured from the image with a known distance
      fps: scalar conversor
      matrix: Homography matrix
      out_image: Im supplied, speed will be written in the centroid position
      n_frames: if supplied, frames previous to the object appearence will have 0 speed
    
    """
    
    for kfilt in filters:
        # speed estimator
        try:

            trans_motion = np.dot(matrix, np.append(kfilt['motion'],np.zeros(1)) )
            dist = np.linalg.norm(trans_motion)
            
            #pix/frame-> m/s
            speed= dist*meter_pix*fps*3.6
            
            sp[str(kfilt['id'])].append(speed)
            if out_image is not None:
        
                cv2.putText(out_image, str(sp), (kfilt['centroid'][0],kfilt['centroid'][1])
                       , cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255)

        except KeyError:

            sp[str(kfilt['id'])] = n_frames*[0]
   
    if out_image is not None: 
        return out_image
    else:
        return 
