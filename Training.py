#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:04:16 2020

@author: brad
"""
# prerequisites
import torch
import numpy as np
from sklearn import svm
from torchvision import datasets, transforms
from torchvision.utils import save_image


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
import config 

config.init()

from config import numcolors, args
global numcolors
from mVAE import train, test, vae,  thecolorlabels, optimizer



#define color labels 
#this list of colors is randomly generated at the start of each epoch (down below)

#numcolors indicates where we are in the color sequence 
#this index value is reset back to 0 at the start of each epoch (down below)
numcolors = 0
#this is the amount of variability in r,g,b color values (+/- this amount from the baseline)

#these define the R,G,B color values for each of the 10 colors.  
#Values near the boundaries of 0 and 1 are specified with colorrange to avoid the clipping the random values
    





for epoch in range(1, 201):
    
    print(numcolors)
    #modified to include color labels
    train(epoch,'iterated')
    colorlabels = np.random.randint(0,10,100000)#regenerate the list of color labels at the start of each test epoch
    numcolors = 0
    if epoch % 5 ==0:
        test('all')  
   
    if epoch in [1, 25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{args.dir}/checkpoint_threeloss_singlegrad'+str(epoch)+'.pth')






