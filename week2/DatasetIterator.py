import os
import glob
import numpy as np
from PIL import Image

def DataIterator(test_path, gt_path, test_prefix='', gt_prefix='',
              test_format='png', gt_format='png', start = 0, end = -1, 
         process_gt = lambda img: img, id_format = lambda form: form, im_proc = lambda proc : proc):
    
    """
    Iterator over the desired slice of the data
    :param test_path: (str) relative or absolute path to the test results images
    :param gt_path: (str) relative or absolute path to the ground truth images
    :param test_prefix: (str) prefix of the test files before their ID (e.g.
    test_A_001235.png has test_A_ as prefix)
    :param gt_prefix: (str) prefix of the ground truth files before their ID
    (e.g. gt001235.png has gt as prefix)
    :param test_format: (str) format of the test images
    :param gt_format: (str) format of the ground truth images
    :param start: (int) Id of the first element of the sequence
    :param start: (int) Id of the last element of the sequence
    :param process_gt: function to be applied to the grountruth for obtaining a specific format
    :param id_format: function to be applied to the start and end in order to find them on the dataset
    :yields: (tuple) Pair of Image - GrounTruth
        -Image
        -Ground Truth
    """
    
    test_files = glob.glob(os.path.join(test_path, test_prefix + '*.' + test_format))
    gt_files = glob.glob(os.path.join(gt_path, gt_prefix + '*.' + gt_format))

    if len(test_files) == 0:
        print ("No images found!")
        return
    
    if len(gt_files) == 0:
        print ("No GT found!")
        return    

    pre_id_imgs = os.path.join(test_path, test_prefix)
    pre_id_gt = os.path.join(gt_path, gt_prefix)

    indices_im = np.array([filename.replace(pre_id_imgs, '').replace('.' + test_format,'') for filename in test_files])
    indices_gt = np.array([filename.replace(pre_id_gt, '').replace('.' + gt_format,'') for filename in gt_files])

    common_id = np.in1d(indices_im , indices_gt)

    indices_im = indices_im[common_id]

    ini = np.where(indices_im == id_format(start))
    end = np.where(indices_im == id_format(end))

    if(len(ini[0])):
        ini = ini[0][0]
    else:
        print ("Couldn't find first element in the dataset. Starting from the beggining")
        ini = 0

    if(len(end[0])):
        end = end[0][0]
    else:
        print ("Couldn't find last element of the sequence in the dataset. Ending at the last element")
        end = -1
    
    if end < ini and end > -1:
        print("last element comes before than the first. Inverting them. Possible corruption?")
        ini, end = end, ini
             
    indices_im = indices_im[ini:end]

    for Im_p, gt_p in [(pre_id_imgs + ind + '.' + test_format, pre_id_gt + ind + '.' + gt_format) for ind in indices_im]:

            pil_img_test = Image.open(Im_p)
            img_test = proc(np.array(pil_img_test))
            
            pil_img_gt = Image.open(gt_p)
            real_img_gt = process_gt(np.array(pil_img_gt))
            
            yield(img_test, real_img_gt)
