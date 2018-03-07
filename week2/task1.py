from data import cdnet
from video import bg_subtraction
from evaluation import metrics
from evaluation import animations
import numpy as np
import matplotlib.pyplot as plt

# Ground truth labels

LABEL_STATIC = 0
LABEL_SHADOW = 50
LABEL_OUTSIDE_ROI = 85
LABEL_UNKWN_MOTION = 170
LABEL_MOTION = 255

OUT_VIDEO_PATH = 'animations/'
OUT_VIDEO_EXTENSION = 'gif' #['gif','mp4']
OUT_VIDEO_CODEC = None #[None,'mp4v']

train_ini_frame = 1050
train_end_frame = 1200

test_ini_frame = 1200
test_end_frame = 1350

HIGHWAY_DATASET = 'highway'
FALL_DATASET = 'fall'
TRAFFIC_DATASET = 'traffic'

def run():

    DATASET = HIGHWAY_DATASET
    # Fit the model to the first half of the images
    ims_train = cdnet.read_dataset(DATASET, train_ini_frame, train_end_frame, colorspace='gray',
                                   annotated=False)

    model = bg_subtraction.create_model(ims_train)

    plt.title('Mean image '+DATASET+' dataset (frames '+str(
        train_ini_frame)+'-'+
              str(train_end_frame)+')')
    plt.imshow(model[0][:, :, 0])
    plt.show()
    plt.title('Variance image '+DATASET+' dataset (frames '
                                                ''+str(train_ini_frame)+'-'+
              str(train_end_frame)+')')
    plt.imshow(model[1][:,:,0])
    plt.show()

    # Test the model with the second half of the images
    ims, gts = cdnet.read_dataset('highway', test_ini_frame, test_end_frame,
                                  colorspace='gray', annotated=True, bg_th=LABEL_SHADOW, fg_th=LABEL_MOTION)
    results_list = []
    results_dicts = []
    alpha_list = [0,1,2,3,4,5,6,7,8,9]

    for alpha in alpha_list:

        pred = bg_subtraction.predict(ims, model, alpha)

        out_filename = 'non_adapt-'+DATASET+'-alpha_' + str(alpha)
        is_mask = True

        if out_filename is not None and OUT_VIDEO_EXTENSION is not None:
            print('>>>>Recording animation in '+OUT_VIDEO_PATH+out_filename+'.'+
                                                OUT_VIDEO_EXTENSION+'...')

            animations.video_recorder(pred, OUT_VIDEO_PATH, out_filename,
                OUT_VIDEO_CODEC, OUT_VIDEO_EXTENSION)

        # Extract metrics (TP, FP, ...) and plot results
        results = metrics.eval_from_mask(pred, gts[:,0], gts[:,1])
        results_list.append(results)

    results_dicts.append(dict(title='Test A (Gaussian Modelling)',
                          data=results_list))

    metrics.plot_results_by_some_param(results_dicts, alpha_list, 'alpha',
                                       'prec-rec-f1')






