from video import bg_subtraction
from evaluation import metrics
from data import cdnet
from video import shadow_detection


def run(dataset):

    train = cdnet.read_sequence('week3', dataset, 'train', colorspace='gray',
                                annotated=False)
    test, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='gray', annotated=True, bg_th = 0)
    test_c, gt = cdnet.read_sequence('week3', dataset, 'test',
                                   colorspace='rgb', annotated=True, bg_th = 0)


    alpha = 1.5
    rho = 0.1



    # Adaptive model prediction
    model = bg_subtraction.create_model(train)
    PredNoShadow = bg_subtraction.predict(test, model, alpha, rho)

    results_no_shadow = metrics.eval_from_mask(pred, gt[:,0], gt[:,1])
    test2 = {'description': 'normal', 'data' : results_no_shadow}
    dic_list = [test2]

    for t1 in np.arange(0.045, 0.075, 0.001):
        for t2 in np.arange(0.0008, 0.018, 0.001):
            shad = shadow_detection.shadow_batch(test_c, t1, t2)
            shadow_filter = np.logical_and(pred, shad)

        dic_list.append({'description': (alpha,beta), 'data' : metrics.eval_from_mask(shadow_filter, gt[:,0], gt[:,1])})

    metrics.summarize_tests(dic_list)
