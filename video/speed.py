import numpy as np
import cv2
import matplotlib.pyplot as plt
    
def speed(sp={}, filters = None, meter_pix = 12.19/20.55  , fps=30, matrix = np.eye(3), out_image = None, 
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
                
                imp1 = np.dot(matrix, np.append(kfilt['centroid'], np.ones(1)))
                imp2  = np.dot(matrix, np.append(sp[str(kfilt['id'])], np.ones(1)))
                imp1/=imp1[2]
                imp2/=imp2[2]               
                trans_motion = imp1-imp2                
                #trans_motion = np.dot(matrix, np.append(kfilt['centroid']-sp[str(kfilt['id'])],np.zeros(1)) )
                                                    
                print(trans_motion)
                #if(trans_motion[2]!=0):
                    #trans_motion/=trans_motion[2]
                    
                dist = np.linalg.norm(trans_motion[:2])
                print('pix/frame: ', dist/skip_frames)
                #pix/frame-> m/s
                speed= (dist/skip_frames)*meter_pix*fps*3.6
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

def plot_speed(speed_dict):
    '''
    Plots the stored speed values in the corresponding dict.
    Params:
        speed_dict: Dictionary generated from the result of tracking 
                    and estimating speed via speed function    
    '''
    for key in sp.keys():
        if(key[0]=='s'):
            
            values = np.array(sp[key])
            values = values[abs(values - np.mean(values)) < 1.2 * np.std(values)]
            plt.plot(np.array(values), label=key)
            print(key, ": mean ", np.mean(values), " median: " , np.median(values))

    plt.legend()
    plt.title('Speed Estimation')
    plt.ylabel('speed')
    plt.xlabel('frame')
    plt.show()  
