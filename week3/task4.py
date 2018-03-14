from video import bg_subtraction
from evaluation import metrics
from data import cdnet
from video import shadow_detection
import cv2


def run(dataset):

    train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True, bg_th = 50)
    test_c, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='rgb', annotated=True, bg_th = 50)

    alpha = {'highway':2.8, 'fall':2.5, 'traffic': 2.25}
    rho = {'highway': 0.20, 'fall': 0.10, 'traffic': 0.15}


    if dataset == 'highway':
        bsize = 50

        se_close = (15, 15)
        se_open = (4, 17)

    elif dataset == 'fall':
        bsize = 800

        se_close = (15, 15)
        se_open = (10, 10)

    elif dataset == 'traffic':
        bsize = 400

        se_open = (10, 10)    

    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    pred = bg_subtraction.predict(test, model, alpha[dataset], rho[dataset])

    results_no_shadow = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
    test2 = {'description': 'normal', 'data' : results_no_shadow}

    dic_list = [test2]

    filled8 = morphology.imfill(pred, neighb=8)
    clean = morphology.filter_small(filled8, bsize, neighb=4)

    if dataset == 'highway' or dataset == 'fall':
        #CLOSING
        st_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            se_close)
        morph_c = morphology.filter_morph(clean, cv2.MORPH_CLOSE,
                                          st_elem)

        #OPENING
        st_elem = cv2.getStructuringElement(cv2.MORPH_RECT, se_open)
        morph = morphology.filter_morph(morph_c, cv2.MORPH_OPEN,
                                        st_elem)

    for t1 in np.arange(0.045, 0.075, 0.005):
        for t2 in np.arange(0.0008, 0.018, 0.01):


            shad = shadow_detection.shadow_batch(test_c, t1, t2)
            shadow_filter = np.logical_and(pred, shad)

        dic_list.append({'description': (t1, t2), 'data' : metrics.eval_from_mask(shadow_filter, gt[:,0], gt[:,1])})

    metrics.summarize_tests(dic_list)    

