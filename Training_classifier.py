
#this Function train 4 classifiers per model: shape labels on shape map, shape labels on color map, color labels on color map and color labels on shape map.

colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]


# prerequisites
import glob, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import config 
from IPython.display import Image, display
import pickle
import cv2

config.init()

from config import numcolors
global numcolors

from mVAE import *

from joblib import dump, load
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION

numModels=10  #number of models that classifiers are trained on

# training classifiers against 10 trained models
for outs in range(1,numModels+1):

    load_checkpoint('output{num}/checkpoint_threeloss_singlegrad200.pth'.format(num=outs))
    print('Training two classifiers based on shape for model number {x}'.format(x=outs))
    numcolors = 0
    classifier_shape_train('noskip')
    
    #save the classifiers in the corresponding folder
    dump(clf_sc, 'output{num}/sc{num}.joblib'.format(num=outs))
    dump(clf_ss, 'output{num}/ss{num}.joblib'.format(num=outs))
    
    print('Training two classifiers based on color for model number {x}'.format(x=outs))
    numcolors = 0
    classifier_color_train('noskip')
    
    #save classifiers in the corresponding folder
    dump(clf_cc, 'output{num}/cc{num}.joblib'.format(num=outs))
    dump(clf_cs, 'output{num}/cs{num}.joblib'.format(num=outs))



    
  
