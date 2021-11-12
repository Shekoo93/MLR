
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


for epoch in range(1, 201):
    
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
        torch.save(checkpoint,f'{args.dir}/checkpoint_threeloss_singlegrad'+str(epoch)+'.pth') #save the model in the directory






