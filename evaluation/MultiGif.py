import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Batch:
    
    def __init__(self, images, title, position):
        self.images = images
        self.name = title
        self.pos = position
        
def make_gif(BatchList, Size):
    
    """
    Function make_gif. Recieves a list of Batch objects containing the image data
    and the total size of the grid. Returns a list of matplot images, in order to make 
    the gif. Hence synchronization might not be an issue no more.
    Params:
    batch_List
    
    """
    
    if(not len(BatchList)):
        print("no batches passed!")
        return
    
    if(not type(BatchList[0]) == Batch):
        print("must pass a list of batch")
        
    n_images = len(BatchList[0].images)
    
    if (not len(Size) == 2 or not type(Size[0]) == int or not type(Size[1]) == int):
        print("incorrect input size")
        return
    
    for batch in BatchList:
        if(n_images != len(batch.images)):
            
            print("each batch must have the same number of images")
            return
        
        if (not len(batch.pos) == 2 or not type(batch.pos[0]) == int 
            or not type(batch.pos[1]) == int or batch.pos[0] < 0 or batch.pos[1] < 0):

            print("Image position for batch " + batch.title + 
                  " must be a tuple of integers")
            return
        
        if(batch.pos[0]**Size[1] + batch.pos[1] > Size[0]*Size[1]):
            print("batch " + batch.title + " position out of bounds")
            return
        
        #Define the position of the images in the flattened vector
        batch.pos = batch.pos[0]*Size[1] + batch.pos[1]
        
    #### All comprovations have been done! More could be added

    #Define the grid for the plottings 
    f, axarr = plt.subplots(Size[0], Size[1], figsize=(15,15))
    
    ## shutdown axes
    axes = axarr.flatten()
    [ax.axis('off') for ax in axes]
    
    #get Size of the resulting image
    width, height = f.get_size_inches() * f.get_dpi()
    width, height= int(width), int(height)

    #define the output array
    image_list = np.zeros((len(BatchList[0].images),height, width, 3), dtype='uint8')
    
    canvas = FigureCanvas(f)
    
    #Maybe this could be speeded up
    for i in range(0, len(BatchList[0].images)):
        
        for batch in BatchList:
        
            axes[batch.pos].imshow(batch.images[i])
            axes[batch.pos].set_title(batch.name)
        
        
        canvas.draw() 
        image_list[i] = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
    
    return image_list
