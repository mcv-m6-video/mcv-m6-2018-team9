import os
import glob
import numpy as np
from PIL import Image

class Dataset(object):
    
    def __init__(self, name, imgfold, gtfold, imgpref, gtpref, 
                 imgformat, gtformat, idformat = lambda x: x):
        self.Name = name
        self.Imagefolder = imgfold
        self.GTfolder = gtfold
        self.ImgPrefix = imgpref
        self.GtPrefix  = gtpref
        self.Imgformat = imgformat
        self.Gtformat = gtformat
        self.Idformat = idformat


def id_format(x):
    return str(int(x)).zfill(6)

TestPrefix='in'
GTPrefix='gt'
TestFormat='jpg'
GTFormat='png'


dataset = 'fall'      
GTFolder = os.path.join('..','datasets', dataset,'groundtruth')
TestFolder = os.path.join('..','datasets', dataset, 'input')

fall = Dataset('fall', TestFolder, GTFolder, TestPrefix, GTPrefix, TestFormat, GTFormat, id_format)

dataset = 'highway'      
gt_folder = os.path.join('..','datasets', dataset,'groundtruth')
tests_folder = os.path.join('..','datasets', dataset, 'input')

highway = Dataset('highway', TestFolder, GTFolder, TestPrefix, GTPrefix, TestFormat, GTFormat, id_format)

dataset = 'traffic'      
gt_folder = os.path.join('..','datasets', dataset,'groundtruth')
tests_folder = os.path.join('..','datasets', dataset, 'input')

traffic = Dataset('highway', TestFolder, GTFolder, TestPrefix, GTPrefix, TestFormat, GTFormat, id_format)

Datasets = {"fall": fall, "highway": highway, "traffic":traffic}


def DataIterator(Dataset, Start = 0, End = -1, 
                 process_img = lambda img : img, process_gt = lambda gt: gt):
    """
    Iterator over the desired slice of the data
    :param Dataset (str or Dataset) Name of the dataset to be used or individual dataset defined
                    as instance of the class Dataset
    :param Start: (int) Id of the first element of the sequence
    :param End: (int) Id of the last element of the sequence
    :param process_img: function to be applied to the images for obtaining a specific format
    :param process_gt: function to be applied to the grountruth for obtaining a specific format

    :yields: (tuple) Pair of Image - GrounTruth
        -Image
        -Ground Truth
    """    
    
    try:
        
        TestPath = Datasets[Dataset].Imagefolder
        GtPath = Datasets[Dataset].GTfolder
        TestPrefix = Datasets[Dataset].ImgPrefix
        GTPrefix = Datasets[Dataset].GtPrefix
        TestFormat = Datasets[Dataset].Imgformat
        GTFormat = Datasets[Dataset].Gtformat
        id_format = Datasets[Dataset].Idformat
        
    except KeyError:
            
            if type(dataset) == Dataset:
                
                    TestPath = dataset.Imagefolder
                    GtPath = dataset.GTfolder
                    TestPrefix = dataset.ImgPrefix
                    GTPrefix = dataset.GtPrefix
                    TestFormat = dataset.Imgformat
                    GTFormat = dataset.Gtformat
                    id_format = dataset.Idformat
            else:
                
                print("Couldn't find dataset!")
                return
    
    
    #Get all files in each directory    
    TestFiles = glob.glob(os.path.join(TestPath, TestPrefix + '*.' + TestFormat))
    GTFiles = glob.glob(os.path.join(GtPath, GTPrefix + '*.' + GTFormat))

    if len(TestFiles) == 0:
        print ("No images found!")
        return
    
    if len(GTFiles) == 0:
        print ("No GT found!")
        return    

    #We can avoid the suposition that images and gt are ordered or that all files can be found
    #in each folder by taking the intersection of their ids
    PreIdImg = os.path.join(TestPath, TestPrefix)
    PreIdGT = os.path.join(GtPath, GTPrefix)

    IndicesImg = np.array([filename.replace(PreIdImg, '').replace('.' + TestFormat,'') 
                           for filename in TestFiles])
    
    IndicesGT = np.array([filename.replace(PreIdGT, '').replace('.' + GTFormat,'') 
                           for filename in GTFiles])

    common_id = np.in1d(IndicesImg , IndicesGT)

    #Get common indices in image folder and GT folder
    IndicesImg = IndicesImg[common_id]

    #filter indices between init and end 
    Start = np.where(IndicesImg == id_format(Start))
    End = np.where(IndicesImg == id_format(End))

    if(len(Start[0])):
        Start = Start[0][0]
    else:
        print ("Couldn't find first element in the dataset. Starting from the beggining")
        Start = 0

    if(len(End[0])):
        End = End[0][0]
    else:
        print ("Couldn't find last element of the sequence in the dataset. Ending at the last element")
        End = -1
    
    if End > -1 and End < Start:
        print("last element comes before than the first. Inverting them. Possible corruption?")
        Start, End = End, Start
    
    #Get target elements               
    IndicesImg = IndicesImg[Start:End]

    #Reform path's with common indices
    for ImgPath, GTPath in [(PreIdImg + Ind + '.' + TestFormat, PreIdGT + Ind + '.' + GTFormat) 
                            for Ind in IndicesImg]:
            
            PilImgTest = Image.open(ImgPath)
            ImgTest = process_img(np.array(PilImgTest))
            
            PilImgGT = Image.open(GTPath)
            GT = process_gt(np.array(PilImgGT))
            
            yield(ImgTest, GT)