from data import cdnet
from video import bg_subtraction


def run():

    # Fit the model to the first half of the images
    ims_train = cdnet.read_dataset('highway', 1, 1200, colorspace='gray',
                                   annotated=False)
    model = bg_subtraction.create_model(ims_train)

    # Test the model with the second half of the images
    ims, gts = cdnet.read_dataset('highway', 1200, 1350,
                                  colorspace='gray', annotated=True)
    pred = bg_subtraction.predict(ims, model, 1)

    import pdb; pdb.set_trace()

    # Extract metrics (TP, FP, ...) and plot results
