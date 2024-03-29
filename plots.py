

#MNIST VAE retreived from https://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb

# Modifications:
#Colorize transform that changes the colors of a grayscale image
#colors are chosen from 10 options:
colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]
#specified in "colorvals" variable below
#also there is a skip connection from the first layer to the last layer to enable reconstructions of new stimuli
#and the VAE bottleneck is split, having two different maps
#one is trained with a loss function for color only (eliminating all shape info, reserving only the brightest color)
#the other is trained with a loss function for shape only

# prerequisites
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
import cv2
from PIL import ImageFilter
import imageio, time
import math
import sys
import pandas as pd
config.init()
from config import numcolors
global numcolors, colorlabels
from PIL import Image
from mVAE import *
from tokens_capacity import *
import os
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION

modelNumber= 1 #which model should be run, this can be 1 through 10
load_checkpoint('output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))
print('Loading the classifiers')
clf_shapeS=load('output{num}/ss{num}.joblib'.format(num=modelNumber))
clf_shapeC=load('output{num}/sc{num}.joblib'.format(num=modelNumber))
clf_colorC=load('output{num}/cc{num}.joblib'.format(num=modelNumber))
clf_colorS=load('output{num}/cs{num}.joblib'.format(num=modelNumber))
#write to a text file
outputFile = open('outputFile.txt'.format(modelNumber),'w')

#Parameters
bs_testing = 1000     # number of images for testing. 20000 is the limit
shape_coeff = 1       #cofficient of the shape map
color_coeff = 1       #coefficient of the color map
l1_coeff = 1          #coefficient of layer 1
l2_coeff = 1          #coefficient of layer 2
shapeLabel_coeff= 1   #coefficient of the shape label
colorLabel_coeff = 1  #coefficient of the color label
bpsize = 2500         #size of the binding pool
token_overlap = .4
bpPortion = int(token_overlap *bpsize) # number binding pool neurons used for each item
normalize_fact_familiar=1              #factors that multiply the BP activations based on whether they are familiar or not (these factors are set to 1 in our model, but can be adjusted to different values)
normalize_fact_novel=1    
numModels=10          #number of models being tested in some of the functions
all_imgs = []

#number of repetions for statistical inference
hugepermnum=10000 
bigpermnum = 500
smallpermnum = 100

#flags that determine which figure/table is simulated
Fig1SuppFlag =0      #reconstructions straight from the VAE (supplementary figure 1)
Fig2aFlag = 0        #binding pool reconstructions
Fig2bFlag = 0        #novel reconstructions
Fig2cFlag = 0        #token reconstructions (reconstructing multiple items)
bindingtestFlag = 0  #simulating binding shape-color of two items
Tab1Flag_noencoding = 0 #classify reconstructions (no memory)
Tab1Flag =0             #classify binding pool memories
Tab1SuppFlag =0        #memory of labels (this is table 1 + Figure 2 in supplemental which includes the data in Figure 3)
Tab2Flag = 1            #Cross correlations for familiar vs novel
noveltyDetectionFlag=0  #detecting whether a stimulus is familiar or not
latents_crossFlag = 0   #Cross correlations for familiar vs novel for when infromation is stored from the shape/color maps vs. L1. versus straight reconstructions 
                        #(This Figure is not included in the paper)
#generate some random samples
zc=torch.randn(64,8).cuda()*1
zs=torch.randn(64,8).cuda()*1
with torch.no_grad():
    sample = vae.decoder_noskip(zs,zc,0).cuda()
    sample_c= vae.decoder_noskip(zs*0,zc,0).cuda()
    sample_s = vae.decoder_noskip(zs, zc*0, 0).cuda()
sample=sample.view(64, 3, 28, 28)
sample_c=sample_c.view(64, 3, 28, 28)
sample_s=sample_s.view(64, 3, 28, 28)
save_image(sample[0:8], 'output{num}/sample.png'.format(num=modelNumber))
save_image(sample_c[0:8], 'output{num}/sample_color.png'.format(num=modelNumber))
save_image(sample_s[0:8], 'output{num}/sample_shape.png'.format(num=modelNumber))

test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))    #combine fashion-mnist and mnist
test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True, num_workers=nw)  #load the dataset

########################## straight reconstrcutions from the VAE (Fig1 supplemental)#########################
if Fig1SuppFlag ==1:
    print('showing reconstructions from shape and color')
    numimg=10
    bs_testing = numimg #number of images to display in this figure
    
    #build a combined dataset out of MNIST and Fasion MNIST
    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))
    test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True, num_workers=nw)
    numcolors = 0
    colorlabels = np.random.randint(0, 10, 100000)
    test_colorlabels = thecolorlabels(test_dataset)
    images, labels = next(iter(test_loader_smaller))
    orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
    imgs = orig_imgs.clone()

    #run them all through the encoder
    l1_act, l2_act, shape_act, color_act = activations(imgs)  #get activations from this small set of images

    #now reconstruct them directly from the VAE
    bothRecon = vae.decoder_noskip(shape_act, color_act, 0).cuda()  # reconstruction directly from the bottleneck (Shape and color)
    shapeRecon= vae.decoder_shape(shape_act, color_act, 0).cuda()  # reconstruction directly from the bottleneck (shape map)
    colorRecon= vae.decoder_color(shape_act, color_act, 0).cuda()  # reconstruction directly from the bottleneck (color map)
    
    #saves the images
    save_image(
        torch.cat([imgs[0: numimg].view(numimg, 3, 28, 28), bothRecon[0: numimg].view(numimg, 3, 28, 28),
                   shapeRecon[0: numimg].view(numimg, 3, 28, 28), colorRecon[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure_disen_VAErecons.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )
######################## Figure 2a #######################################################################################
#store items using both features, and separately color and shape (memory retrievals)
if Fig2aFlag==1:
    print('generating figure 2a, reconstructions from the binding pool')
    numimg= 6 #number of images to display in this figure
    numcolors = 0
    colorlabels = np.random.randint(0, 10, 100000)
    colorlabels = thecolorlabels(test_dataset)
    bs_testing = numimg # 20000 is the limit
    test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True, num_workers=nw)
    images, shapelabels = next(iter(test_loader_smaller))#peel off a large number of images
    orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
    imgs = orig_imgs.clone()
    colorlabels = colorlabels[0:bs_testing]

    #run them all through the encoder
    l1_act, l2_act, shape_act, color_act = activations(imgs)  #get activations from this small set of images

    #binding pool outputs
    BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk =            BP(bpPortion , l1_act, l2_act, shape_act, color_act, shape_coeff, color_coeff,l1_coeff,l2_coeff,normalize_fact_familiar)
    BP_in, shape_out_BP_shapeonly,  color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion , l1_act, l2_act, shape_act, color_act, shape_coeff, 0,0,0,normalize_fact_familiar)
    BP_in,  shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion , l1_act, l2_act, shape_act, color_act, 0, color_coeff,0,0,normalize_fact_familiar)
    BP_in,  shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion , l1_act, l2_act, shape_act, color_act, 0, 0,l1_coeff,0,normalize_fact_familiar)
    BP_in,  shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion , l1_act, l2_act, shape_act, color_act, 0, 0,0,l2_coeff,normalize_fact_familiar)

    #memory retrievals from Bottleneck storage
    bothRet = vae.decoder_noskip(shape_out_BP_both, color_out_BP_both, 0).cuda()  # memory retrieval from the bottleneck
    shapeRet = vae.decoder_shape(shape_out_BP_shapeonly, color_out_BP_shapeonly , 0).cuda()  #memory retrieval from the shape map
    colorRet = vae.decoder_color(shape_out_BP_coloronly, color_out_BP_coloronly, 0).cuda()  #memory retrieval from the color map
    
    #save images
    save_image(
        torch.cat([imgs[0: numimg].view(numimg, 3, 28, 28), bothRet[0: numimg].view(numimg, 3, 28, 28),
                   shapeRet[0: numimg].view(numimg, 3, 28, 28), colorRet[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_BP_bottleneck_.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )
    
    #memory retrievals when information was stored from L1 and L2
    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 1, 'noskip') #bp retrievals from layer 1
    BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 2, 'noskip') #bp retrievals from layer 2

    save_image(
        torch.cat([
                   BP_layer2_noskip[0: numimg].view(numimg, 3, 28, 28), BP_layer1_noskip[0: numimg].view(numimg, 3, 28, 28)], 0),
        'output{num}/figure2a_layer2_layer1.png'.format(num=modelNumber),
        nrow=numimg,
        normalize=False,
        range=(-1, 1),
    )
############################# Figure 2b#################################################################################
if Fig2bFlag==1:
    print('generating Figure 2b, Novel characters retrieved from memory of L1 and Bottleneck')
    numimg = 6   
    trans2 = transforms.ToTensor()
    
    #loads the novel stimuli (Bengali charachters)
    for i in range (1,numimg+1):
        img = Image.open('{each}_thick.png'.format(each=i))
        img = np.mean(img,axis=2)
        img[img < 64] = 0   #necessary for correcting baseline value of non thick stimuli

        img_new = Colorize_func(img)
        image = trans2(img_new)*1.3
        all_imgs.append(image)
    all_imgs = torch.stack(all_imgs)
    imgs = all_imgs.view(-1, 3 * 28 * 28).cuda()
    
    #save images
    save_image(
            torch.cat([trans2(img_new).view(1, 3, 28, 28)], 0),
            'output{num}/figure10test.png'.format(num=modelNumber),
            nrow=numimg,
            normalize=False,
            range=(-1, 1),
        )
    
    #increasing the difference between positive and zero-level activations in L1 using the follwoing transformation:
    l1_act, l2_act, shape_act, color_act = activations(imgs)
    l1_act_tr = l1_act.clone()
    l1_act_tr[l1_act!=0] = l1_act_tr[l1_act!=0] + 2
    l1_act_tr[l1_act_tr == 0] = -3

    # BP outputs of L1, L2 shape and color maps
    BP_in, shape_out_BP, color_out_BP, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act_tr, l2_act, shape_act, color_act, shape_coeff, color_coeff,0,0,normalize_fact_novel)
    BPRep, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act_tr, l2_act, shape_act, color_act, 0, 0,l1_coeff,0,normalize_fact_novel)
    BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act_tr, l2_act, shape_act, color_act, 0, 0,0,l2_coeff,normalize_fact_novel)
    BP_layerI_out = BP_layerI_out.squeeze()
    BP_layerI_out[BP_layerI_out < 0] = 0

    #reconstruct directly from activation without being stored in BP
    recon_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act, l2_act, 1, 'skip')
    
    #reconstruct directly from layer 1 skip after stored in BP
    BP_layer1_skip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out,1, 'skip')

    #reconstruct directly from layer 1 noskip  (i.e. through the bottleneck)
    BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,BP_layer2_out, 1, 'noskip')

    #save images
    save_image(
            torch.cat([imgs[0: numimg].view(numimg, 3, 28, 28), recon_layer1_skip[0: numimg].view(numimg, 3, 28, 28),
                       BP_layer1_skip[0: numimg].view(numimg, 3, 28, 28),BP_layer1_noskip[0: numimg].view(numimg, 3, 28, 28) ], 0),
            'output{num}/figure2b.png'.format(num=modelNumber),
            nrow=numimg,
            normalize=False,
            range=(-1, 1),
        )
#####################################Figure 2c (This part is token-related) #########################################################################3
if Fig2cFlag ==1 :
    print('generating Figure 2c. Storing and retrieving multiple items')
    
    numimg= 100 #number of images to display in this figure
    bs_testing = numimg

    #build a combined dataset out of MNIST and Fasion MNIST
    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))
    test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True, num_workers=nw)

    numcolors = 0
    colorlabels = np.random.randint(0, 10, 100000)
    test_colorlabels = thecolorlabels(test_dataset)
    images, labels = next(iter(test_loader_smaller))
    orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
    imgs = orig_imgs.clone()
    l1_act, l2_act, shape_act, color_act = activations(imgs)
    storelabels=0
    
    
    for n in range(1,5):
        
        #generating one hot labels as the function's input
        oneHotShape=torch.zeros(n,20)
        oneHotcolor=torch.zeros(n,10)
        
        #activations after shape/color map was stored in the BP
        shape_out_all, color_out_all, l2_out_all, l1_out_all, shapelabel_junk, colorlabel_junk = BPTokens_with_labels(bpsize, bpPortion,storelabels, shape_coeff, color_coeff, shape_act, color_act,
                                                                        l1_act, l2_act,oneHotShape, oneHotcolor,n, 0, normalize_fact_familiar)
                                                                        
        retrievals = vae.decoder_noskip(shape_out_all, color_out_all, 0).cuda()  
        imgs = orig_imgs.clone()
        save_image(imgs[0: n].view(n, 3, 28, 28), 'output{num}/figure2coriginals{d}.png'.format(num=modelNumber,d=n))
        save_image(retrievals[0: n].view(n, 3, 28, 28), 'output{num}/figure2cretrieved{d}.png'.format(num=modelNumber,d=n))
###################Table 2##################################################
if Tab2Flag ==1:

      
    print('Tab2 loss of quality of familiar vs novel items using correlation')
    setSizes=[1,2,3,4] #number of tokens
   
    #list of correlation values and standard errors for each set size
    familiar_corr_all=list()  
    familiar_corr_all_se=list() 
    novel_corr_all=list()
    novel_corr_all_se=list()  
    familiar_skip_all=list()
    familiar_skip_all_se=list()
    novel_BN_all=list()
    novel_BN_all_se = list()
    
    #number of times it repeats storing/retrieval
    perms = bigpermnum
    for numItems in setSizes:
        #list of correlation values across models
        familiar_corr_models = list()  
        novel_corr_models = list()
        familiar_skip_models=list()
        novel_BN_models=list()
        print('SetSize {num}'.format(num=numItems))
        for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
            load_checkpoint(
                'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

            # reset the data set for each set size
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=nw)
            
            #correlation of familiar items as a function of set size when shape/color maps are stored (This function is in tokens_capacity.py)
            familiar_corrValues= storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0,1)
            
            #correlation of familiar items as a function of set size when L1 is stored and items are retrived via the skip
            familiar_corrValues_skip = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                      test_loader_smaller, 'fam', 1, 1)

            #correlation of novel items as a function of set size when L1 is tored and items are retrived via skip
            novel_corrValues = storeretrieve_crosscorrelation_test(numItems, perms, bpsize,
                                                                                             bpPortion, shape_coeff,
                                                                                             color_coeff,
                                                                                             normalize_fact_familiar,
                                                                                             normalize_fact_novel,
                                                                                             modelNumber,
                                                                                             test_loader_smaller, 'nov', 1,1)
            
            #correlation of novel items as a function of set size when shape/color maps are stored                                                                                            
            novel_corrValues_BN = storeretrieve_crosscorrelation_test(numItems, perms, bpsize,
                                                                   bpPortion, shape_coeff,
                                                                   color_coeff,
                                                                   normalize_fact_familiar,
                                                                   normalize_fact_novel,
                                                                   modelNumber,
                                                                   test_loader_smaller, 'nov',
                                                                   0, 1)

            #correlation values across models
            familiar_corr_models.append(familiar_corrValues)
            familiar_skip_models.append(familiar_corrValues_skip)
            novel_corr_models.append(novel_corrValues)
            novel_BN_models.append(novel_corrValues_BN)
        
        #reshaping the correlation values matrix
        familiar_corr_models_all=np.array(familiar_corr_models).reshape(-1,1)
        novel_corr_models_all = np.array(novel_corr_models).reshape(1, -1)
        familiar_skip_models_all=np.array(familiar_skip_models).reshape(1,-1)
        novel_BN_models_all=np.array(novel_BN_models).reshape(1,-1)

        #correlation values for each set size + the corresponding standard errors
        familiar_corr_all.append(np.mean(familiar_corr_models_all))
        familiar_corr_all_se.append(np.std(familiar_corr_models_all)/math.sqrt(numModels))
        novel_corr_all.append(np.mean( novel_corr_models_all))
        novel_corr_all_se.append(np.std(novel_corr_models_all)/math.sqrt(numModels))
        familiar_skip_all.append(np.mean(familiar_skip_models_all))
        familiar_skip_all_se.append(np.std(familiar_skip_models_all)/math.sqrt(numModels))
        novel_BN_all.append(np.mean(novel_BN_models_all))
        novel_BN_all_se.append(np.std(novel_BN_models_all)/math.sqrt(numModels))

    #the mean correlation value between input and recontructed images for familiar items (stored and retrived from shape/color maps)
    outputFile.write('Familiar correlation\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE  {2:.3g}\n'.format(setSizes[i],familiar_corr_all[i],familiar_corr_all_se[i]))

    #the mean correlation value between input and recontructed images for novel items (stored L1, retrived via skip)
    outputFile.write('\nfNovel correlation\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], novel_corr_all[i], novel_corr_all_se[i]))

    #the mean correlation value between input and recontructed images for familiar items (stored L1, retrived via akip)
    outputFile.write('\nfamiliar correlation vis skip \n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], familiar_skip_all[i], familiar_skip_all_se[i]))
    
    #the mean correlation value between input and recontructed images for novel items (stored and retrived from shape/color maps)
    outputFile.write('\nnovel correlation via BN \n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], novel_BN_all[i], novel_BN_all_se[i]))

    #This part (not included in the paper) visualizes the cross correlation between novel shapes retrieved from the skip and familiar shapes retrived from the BN  
    plt.figure()
    familiar_corr_all=np.array(familiar_corr_all)

    novel_corr_all=np.array(novel_corr_all)

    plt.errorbar(setSizes,familiar_corr_all,yerr=familiar_corr_all_se, fmt='o',markersize=3)
    plt.errorbar(setSizes, novel_corr_all, yerr=novel_corr_all_se, fmt='o', markersize=3)

    
    plt.axis([0,6, 0, 1])
    plt.xticks(np.arange(0,6,1))
    plt.show()
#############################################
#comparing the cross correlation between items and their reconstructions when stored into memory vs. when they're not stored (this is not included in the paper)
if latents_crossFlag ==1:
    
    print('cross correlations for familiar items when reconstructed and when retrived from BN or L1+skip ')
    setSizes=[1,2,3,4] #number of tokens
    
    #list of mean correlation values and standard errors for each set size
    noskip_recon_mean=list() 
    noskip_recon_se=list()   
    noskip_ret_mean=list()
    noskip_ret_se=list()
    skip_recon_mean=list()
    skip_recon_se=list()
    skip_ret_mean=list()
    skip_ret_se=list()
    perms = bigpermnum #number of times it repeats storing/retrieval
    for numItems in setSizes: 
      
        #list of correlation values across models
        noskip_Reconmodels=list() 
        noskip_Retmodels=list()
        skip_Reconmodels=list()
        skip_Retmodels=list()
        print('SetSize {num}'.format(num=numItems))
        for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
            load_checkpoint(
                'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))
            # reset the data set for each set size
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=nw)
            # This function is in tokens_capacity.py
            #familiar items reconstrcuted via BN with no memory
            noskip_noMem= storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0, 0)
            # familiar items retrieved via BN
            noskip_Mem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                        color_coeff,
                                                                        normalize_fact_familiar,
                                                                        normalize_fact_novel, modelNumber,
                                                                        test_loader_smaller, 'fam', 0, 1)
            #recon from L1
            skip_noMem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                     test_loader_smaller, 'fam', 1, 0)
             #retrieve from L1 +skip                                                
            skip_Mem = storeretrieve_crosscorrelation_test(numItems, perms, bpsize, bpPortion, shape_coeff,
                                                                      color_coeff,
                                                                      normalize_fact_familiar,
                                                                      normalize_fact_novel, modelNumber,
                                                                      test_loader_smaller, 'fam', 1, 1)
            #appending retrievals/reconstructions
            noskip_Reconmodels.append(noskip_noMem)
            noskip_Retmodels.append(noskip_Mem)
            skip_Reconmodels.append(skip_noMem)
            skip_Retmodels.append(skip_Mem)
            
        #reshaping the arrays 
        noskip_Reconmodels_all=np.array(noskip_Reconmodels).reshape(-1,1)
        noskip_Retmodels_all=np.array(noskip_Retmodels).reshape(-1,1)
        skip_Reconmodels_all = np.array(skip_Reconmodels).reshape(1, -1)
        skip_Retmodels_all=np.array(skip_Retmodels).reshape(1,-1)
        
        #reconstructions and retrievals: mean cross correlation between items and their recons/retrivelas 
        noskip_recon_mean.append(np.mean(noskip_Reconmodels_all))
        noskip_recon_se.append(np.std(noskip_Reconmodels_all)/math.sqrt(numModels))
        noskip_ret_mean.append(np.mean(noskip_Retmodels_all))
        noskip_ret_se.append(np.std(noskip_Retmodels_all) / math.sqrt(numModels))
        skip_recon_mean.append(np.mean(skip_Reconmodels_all))
        skip_recon_se.append(np.std(skip_Reconmodels_all) / math.sqrt(numModels))
        skip_ret_mean.append(np.mean(skip_Retmodels_all))
        skip_ret_se.append(np.std(skip_Retmodels_all) / math.sqrt(numModels))

    #the mean correlation value between input and recontructed images for familiar and novel stimuli
    outputFile.write('correlation for recons from BN\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE  {2:.3g}\n'.format(setSizes[i],noskip_recon_mean[i],noskip_recon_se[i]))

    outputFile.write('\nCorrelation for retrievals from BN\n')
    for i in range(len(setSizes)):
        outputFile.write('SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i],noskip_ret_mean[i],noskip_ret_se[i]))

    outputFile.write('\ncorrelation for recons from skip\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], skip_recon_mean[i], skip_recon_se[i]))

    outputFile.write('\ncorrelation for retrievals from skip\n')
    for i in range(len(setSizes)):
        outputFile.write(
            'SS {0} Corr  {1:.3g}   SE {2:.3g}\n'.format(setSizes[i], skip_ret_mean[i], skip_ret_se[i]))
        
    #plots the correlation values for familiar/BN and novel/L1 retrievals
    plt.figure()
    correlations=np.array([skip_recon_mean,noskip_recon_mean, skip_ret_mean,noskip_ret_mean]).squeeze()
    corr_se=np.array([skip_recon_se,noskip_recon_se, skip_ret_se,noskip_ret_se]).squeeze()
    fig, ax = plt.subplots()
    pos=np.array([1,2,3,4])
    ax.bar(pos, correlations, yerr=corr_se, width=.4, alpha=.6, ecolor='black', color=['blue', 'blue', 'red', 'red'])
    plt.show()
########################  Ability to extract the correct token from a shape-only stimulus######################
if bindingtestFlag ==1:
 
    perms = bigpermnum
    correctToken=np.tile(0.0,numModels)
    correctToken_diff=np.tile(0.0,numModels)
    accuracyColor=np.tile(0.0,numModels)
    accuracyColor_diff=np.tile(0.0,numModels)
    accuracyShape=np.tile(0.0,numModels)
    accuracyShape_diff=np.tile(0.0,numModels)
    for modelNumber in range(1,numModels+1):  
        print('testing binding cue retrieval')      
        bs_testing = 2 
        
        #grey shape cue binding accuracy for only two items when the two items are the same (this funcyion is in Tokens_cpapacity.py)
        correctToken[modelNumber-1],accuracyColor[modelNumber-1],accuracyShape[modelNumber-1] = binding_cue(bs_testing, perms, bpsize, bpPortion, shape_coeff, color_coeff, 'same',
                                                modelNumber)

        # grey shape cue binding accuracy for only two items when the two items are different (this funcyion is in Tokens_cpapacity.py)
        correctToken_diff[modelNumber-1],accuracyColor_diff[modelNumber-1] ,accuracyShape_diff[modelNumber-1] = binding_cue(bs_testing, perms, bpsize, bpPortion, shape_coeff, color_coeff
                                                         , 'diff', modelNumber)
        
    #retriveing correct token + correct shape +correct color and compute the mean and standard deviation across models
    correctToekn_all= correctToken.mean()
    SD=correctToken.std()
    correctToekn_diff_all=correctToken_diff.mean()
    SD_diff=correctToken_diff.std()
    accuracyColor_all=accuracyColor.mean()
    SD_color= accuracyColor.std()   
    accuracyColor_diff_all=accuracyColor_diff.mean()
    SD_color_diff=accuracyColor_diff.std()
    accuracyShape_all=accuracyShape.mean()
    SD_shape= accuracyShape.std()   
    accuracyShape_diff_all=accuracyShape_diff.mean()
    SD_shape_diff=accuracyShape_diff.std()
    
    #write the outputs in the file
    outputFile.write('the correct retrieved token for same shapes condition is: {num} and SD is {sd}'.format(num=correctToekn_all, sd=SD))
    outputFile.write('\n the correct retrieved color for same shapes condition is: {num} and SD is {sd}'.format(num=accuracyColor_all, sd=SD_color))
    outputFile.write('\n the correct retrieved shape for same shapes condition is: {num} and SD is {sd}'.format(num=accuracyShape_all, sd=SD_shape))
    outputFile.write(
        '\n the correct retrieved token for different shapes condition is: {num} and SD is {sd}'.format(num=correctToekn_diff_all, sd=SD_diff))
    outputFile.write(
        '\n the correct retrieved color for different shapes condition is: {num} and SD is {sd}'.format(num=accuracyColor_diff_all, sd=SD_color_diff))
    outputFile.write(
        '\n the correct retrieved shape for different shapes condition is: {num} and SD is {sd}'.format(num=accuracyShape_diff_all, sd=SD_shape_diff))
#############Table 1 for the no memmory condition#####################
#Disentanglemenat of shape and color without storing in memory using classifier accuracies

perms=hugepermnum
if Tab1Flag_noencoding == 1:
    print('Table 1 shape labels predicted by the classifier before encoded in memory')
    SSreport = np.tile(0.0,[perms,numModels])
    SCreport = np.tile(0.0,[perms,numModels])
    CCreport = np.tile(0.0,[perms,numModels])
    CSreport = np.tile(0.0,[perms,numModels])
    for modelNumber in range(1,numModels +1):  # which model should be run, this can be 1 through 10
        load_checkpoint('output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))
        print('doing model {0} for Table 1'.format(modelNumber))
        clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
        clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
        clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
        clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))
        
        #testing the classifiers accuracy on shape/color disentanglement across 10 models and "rep" number of binding pools
        for rep in range(0,perms):
           pred_cc, pred_cs, CCreport[rep,modelNumber - 1], CSreport[rep,modelNumber - 1] = classifier_color_test('noskip',
                                                                                                       clf_colorC,
                                                                                                       clf_colorS)
           pred_ss, pred_sc, SSreport[rep,modelNumber-1], SCreport[rep,modelNumber-1] = classifier_shape_test('noskip', clf_shapeS, clf_shapeC)
    CCreport=CCreport.reshape(1,-1)
    CSreport=CSreport.reshape(1,-1)
    SSreport=SSreport.reshape(1,-1)
    SCreport=SCreport.reshape(1,-1)
    
    #write the outputs in a txt.file
    outputFile.write('Table 1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g}\n'.format(SSreport.mean(),SSreport.std()/math.sqrt(numModels*perms), SCreport.mean(),  SCreport.std()/math.sqrt(numModels) ))
    outputFile.write('Table 1, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g}\n'.format(CCreport.mean(),CCreport.std()/math.sqrt(numModels*perms), CSreport.mean(),  CSreport.std()/math.sqrt(numModels)))
########################## Table 1 for memory conditions ######################################################################
#testing the classifiers accuracy for shape/color maps when ONE ITEM is stored in the BP
if Tab1Flag == 1:
    
    perms=10
    SSreport_both = np.tile(0.0, [perms,numModels])
    SCreport_both = np.tile(0.0, [perms,numModels])
    CCreport_both = np.tile(0.0, [perms,numModels])
    CSreport_both = np.tile(0.0, [perms,numModels])
    SSreport_shape = np.tile(0.0, [perms,numModels])
    SCreport_shape = np.tile(0.0, [perms,numModels])
    CCreport_shape = np.tile(0.0, [perms,numModels])
    CSreport_shape = np.tile(0.0, [perms,numModels])
    SSreport_color = np.tile(0.0, [perms,numModels])
    SCreport_color = np.tile(0.0, [perms,numModels])
    CCreport_color = np.tile(0.0, [perms,numModels])
    CSreport_color = np.tile(0.0, [perms,numModels])
    SSreport_l1 = np.tile(0.0, [perms,numModels])
    SCreport_l1= np.tile(0.0, [perms,numModels])
    CCreport_l1 = np.tile(0.0, [perms,numModels])
    CSreport_l1 = np.tile(0.0, [perms,numModels])
    SSreport_l2 = np.tile(0.0, [perms,numModels])
    SCreport_l2 = np.tile(0.0, [perms,numModels])
    CCreport_l2 = np.tile(0.0, [perms,numModels])
    CSreport_l2 = np.tile(0.0, [perms,numModels])
    
    for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
        load_checkpoint(
            'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))

        print('doing model {0} for Table 1'.format(modelNumber))
        clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
        clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
        clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
        clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))
        print('Doing Table 1')
        for rep in range(0,perms):
            numcolors = 0
            colorlabels = thecolorlabels(test_dataset)
            bs_testing = 1000  # 20000 is the limit
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_testing, shuffle=True,
                                                          num_workers=nw)
            colorlabels = colorlabels[0:bs_testing]
            images, shapelabels = next(iter(test_loader_smaller))  # peel off a large number of images
            orig_imgs = images.view(-1, 3 * 28 * 28).cuda()
            imgs = orig_imgs.clone()
            
            #run them all through the encoder
            l1_act, l2_act, shape_act, color_act = activations(imgs)  # get activations from this small set of images

            # now store and retrieve them from the BP
            BP_in, shape_out_BP_both, color_out_BP_both, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                         shape_act, color_act,
                                                                                         shape_coeff, color_coeff, 1, 1,
                                                                                         normalize_fact_familiar)
            BP_in, shape_out_BP_shapeonly, color_out_BP_shapeonly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                                   l2_act, shape_act,
                                                                                                   color_act,
                                                                                                   shape_coeff, 0, 0, 0,
                                                                                                   normalize_fact_familiar)
            BP_in, shape_out_BP_coloronly, color_out_BP_coloronly, BP_layerI_junk, BP_layer2_junk = BP(bpPortion, l1_act,
                                                                                                   l2_act, shape_act,
                                                                                                   color_act, 0,
                                                                                                   color_coeff, 0, 0,
                                                                                                   normalize_fact_familiar)

            BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_out, BP_layer2_junk = BP(bpPortion, l1_act, l2_act,
                                                                                        shape_act, color_act, 0, 0,
                                                                                        l1_coeff, 0,
                                                                                        normalize_fact_familiar)

            BP_in, shape_out_BP_junk, color_out_BP_junk, BP_layerI_junk, BP_layer2_out = BP(bpPortion, l1_act, l2_act,
                                                                                        shape_act, color_act, 0, 0, 0,
                                                                                        l2_coeff,
                                                                                        normalize_fact_familiar)

           # Table 1: classifier accuracy for shape and color for memory retrievals
            pred_ss, pred_sc, SSreport_both[rep,modelNumber-1], SCreport_both[rep,modelNumber-1] = classifier_shapemap_test_imgs(shape_out_BP_both, shapelabels,
                                                                             colorlabels, bs_testing, clf_shapeS,
                                                                             clf_shapeC)
            pred_cc, pred_cs, CCreport_both[rep,modelNumber-1], CSreport_both[rep,modelNumber-1]= classifier_colormap_test_imgs(color_out_BP_both, shapelabels,
                                                                             colorlabels, bs_testing, clf_colorC,clf_colorS)
          
            #classifiers accuracy for memory retrievals of BN_shapeonly for Table 1
            pred_ss, pred_sc, SSreport_shape[rep,modelNumber - 1], SCreport_shape[
            rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_shapeonly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_shape[rep,modelNumber - 1], CSreport_shape[
            rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_shapeonly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_colorC, clf_colorS)

            #classifiers accuracy for memory retrievals of BN_coloronly for Table 1
            pred_ss, pred_sc, SSreport_color[rep,modelNumber - 1], SCreport_color[
            rep,modelNumber - 1] = classifier_shapemap_test_imgs(shape_out_BP_coloronly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_color[rep,modelNumber - 1], CSreport_color[
            rep,modelNumber - 1] = classifier_colormap_test_imgs(color_out_BP_coloronly, shapelabels,
                                                             colorlabels,
                                                             bs_testing, clf_colorC, clf_colorS)
            # bp retrievals from layer 1
            BP_layer1_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                                BP_layer2_out, 1,
                                                                                                'noskip')  
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            
            #classifiers accuracy for L1 
            pred_ss, pred_sc, SSreport_l1[rep,modelNumber - 1], SCreport_l1[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_l1[rep,modelNumber - 1], CSreport_l1[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                             bs_testing, clf_colorC, clf_colorS)
            
            #bp retrievals from layer 2
            BP_layer2_noskip, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(BP_layerI_out,
                                                                                                BP_layer2_out, 2,'noskip') 
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            
            #classifiers accuracy for L2
            pred_ss, pred_sc, SSreport_l2[rep,modelNumber - 1], SCreport_l2[rep,modelNumber - 1] = classifier_shapemap_test_imgs(z_shape, shapelabels, colorlabels,
                                                                             bs_testing, clf_shapeS, clf_shapeC)
            pred_cc, pred_cs, CCreport_l2[rep,modelNumber - 1], CSreport_l2[rep,modelNumber - 1] = classifier_colormap_test_imgs(z_color, shapelabels, colorlabels,
                                                                             bs_testing, clf_colorC, clf_colorS)
            
    #classifiers accuracies for both shape and color, shape only, color only, L1 and L2
    SSreport_both=SSreport_both.reshape(1,-1)
    SSreport_both=SSreport_both.reshape(1,-1)
    SCreport_both=SCreport_both.reshape(1,-1)
    CCreport_both=CCreport_both.reshape(1,-1)
    CSreport_both=CSreport_both.reshape(1,-1)
    
    SSreport_shape=SSreport_shape.reshape(1,-1)
    SCreport_shape=SCreport_shape.reshape(1,-1)
    CCreport_shape=CCreport_shape.reshape(1,-1)
    CSreport_shape=CSreport_shape.reshape(1,-1)
    
    CCreport_color=CCreport_color.reshape(1,-1)
    CSreport_color=CSreport_color.reshape(1,-1)
    SSreport_color=SSreport_color.reshape(1,-1)
    SCreport_color=SCreport_color.reshape(1,-1)
    
    SSreport_l1=SSreport_l1.reshape(1,-1)
    SCreport_l1=SCreport_l1.reshape(1,-1)
    CCreport_l1=CCreport_l1.reshape(1,-1)
    CSreport_l1=CSreport_l1.reshape(1,-1)
        
    SSreport_l2= SSreport_l2.reshape(1,-1)
    SCreport_l2=SCreport_l2.reshape(1,-1)   
    CCreport_l2= CCreport_l2.reshape(1,-1)
    CSreport_l2=CSreport_l2.reshape(1,-1)
    
   #mean classifiers accuracies across models and repetitions (each repetition generates a random BP)          
    outputFile.write(
        'Table 2 both shape and color, accuracy of SS {0:.4g} SE{1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_both.mean(),
            SSreport_both.std()/math.sqrt(numModels*perms),
            SCreport_both.mean(),
            SCreport_both.std()/math.sqrt(numModels*perms),
            CCreport_both.mean(),
            CCreport_both.std()/math.sqrt(numModels*perms),
            CSreport_both.mean(),
            CSreport_both.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 shape only, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_shape.mean(),
            SSreport_shape.std()/math.sqrt(numModels*perms),
            SCreport_shape.mean(),
            SCreport_shape.std()/math.sqrt(numModels*perms),
            CCreport_shape.mean(),
            CCreport_shape.std()/math.sqrt(numModels*perms),
            CSreport_shape.mean(),
            CSreport_shape.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 color only, accuracy of CC {0:.4g} SE {1:.4g}, accuracy of CS {2:.4g} SE {3:.4g},\n accuracy of SS {4:.4g} SE {5:.4g},accuracy of SC {6:.4g} SE {7:.4g}\n'.format(
            CCreport_color.mean(),
            CCreport_color.std()/math.sqrt(numModels*perms),
            CSreport_color.mean(),
            CSreport_color.std()/math.sqrt(numModels*perms),
            SSreport_color.mean(),
            SSreport_color.std()/math.sqrt(numModels*perms),
            SCreport_color.mean(),
            SCreport_color.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 l1, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l1.mean(),
            SSreport_l1.std()/math.sqrt(numModels*perms),
            SCreport_l1.mean(),
            SCreport_l1.std()/math.sqrt(numModels*perms),
            CCreport_l1.mean(),
            CCreport_l1.std()/math.sqrt(numModels*perms),
            CSreport_l1.mean(),
            CSreport_l1.std()/math.sqrt(numModels*perms)))

    outputFile.write(
        'Table 2 l2, accuracy of SS {0:.4g} SE {1:.4g}, accuracy of SC {2:.4g} SE {3:.4g},\n accuracy of CC {4:.4g} SE {5:.4g}, accuracy of CS {6:.4g} SE {7:.4g}\n'.format(
            SSreport_l2.mean(),
            SSreport_l2.std()/math.sqrt(numModels*perms),
            SCreport_l2.mean(),
            SCreport_l2.std()/math.sqrt(numModels*perms),
            CCreport_l2.mean(),
            CCreport_l2.std()/math.sqrt(numModels*perms),
            CSreport_l2.mean(),
            CSreport_l2.std()/math.sqrt(numModels*perms)))
################################################ storing visual information (shape and color) along with the categorical label###################################
if Tab1SuppFlag ==1:
    print('Table 1S computing the accuracy of storing labels along with shape and color information')
    ftest_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=False,transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]),download=False)
    ftest_dataset.targets= ftest_dataset.targets+10
    test_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=False,transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]),download=False)

    #build a combined dataset out of MNIST and Fasion MNIST
    test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST, ftest_dataset))
    perms = hugepermnum
    setSizes = [1,2,3,4]  # number of tokens
    
    
    #list of classification accuracies (including SE) for shape and color labels when visual infromation and labels are stored together
    totalAccuracyShape = list()
    totalSEshape=list()
    totalAccuracyColor = list()
    totalSEcolor=list()
    
    #list of classification accuracies(including SE) for shape and color visual recons when only visual information is stored (no label)
    totalAccuracyShape_visual=list()
    totalSEshapeVisual=list()
    totalAccuracyColor_visual=list()
    totalSEcolorVisual=list()
    
    #list of classification accuracies (including SE) for shape and color visual recons when visual infromation and labels are stored together
    totalAccuracyShapeWlabels = list()
    totalSEshapeWlabels=list()
    totalAccuracyColorWlabels = list()
    totalSEcolorWlabels=list()
      
    #list of classification accuracies (including SE) for shape and color visual recons when half of the visual infromation and labels are stored together
    totalAccuracyShape_half=list()
    totalSEshape_half=list()
    totalAccuracyColor_half=list()
    totalSEcolor_half=list()
  
    #list of classification accuracies (including SE) for shape and color labels when only labels are stored
    totalAccuracyShape_cat=list()
    totalSEshape_cat=list()
    totalAccuracyColor_cat=list()
    totalSEcolor_cat=list()
    
    #this part was used to create the dot plots that we generated in another code in Panda
    shape_dotplots_models=list() 
    color_dotplots_models=list()
    shapeVisual_dotplots_models=list()
    colorVisual_dotplots_models=list()
    shapeWlabels_dotplots_models=list()
    colorWlabels_dotplots_models=list()
    shape_half_dotplots_models=list()
    color_half_dotplots_models=list()
    shape_cat_dotplots_models=list()
    color_cat_dotplots_models=list()

    for numItems in setSizes:
            print('Doing label/shape storage:  Setsize {num}'.format(num=numItems))
            test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=numItems, shuffle=True,
                                                              num_workers=0)

            #list of mean accuracy across models
            accuracyShapeModels=list()
            accuracyColorModels=list()
            accuracyShapeWlabelsModels=list()
            accuracyColorWlabelsModels=list()
            accuracyShapeVisualModels=list()
            accuracyColorVisualModels=list()
            accuracyShapeModels_half = list()
            accuracyColorModels_half = list()
            accuracyShapeModels_cat = list()
            accuracyColorModels_cat = list()
            for modelNumber in range(1, numModels + 1):  # which model should be run, this can be 1 through 10
                accuracyShape = list()
                accuracyColor = list()
                accuracyShape_visual=list()
                accuracyColor_visual=list()
                accuracyShape_wlabels = list()
                accuracyColor_wlabels = list()
                accuracyShape_half = list()
                accuracyColor_half = list()
                accuracyShape_cat = list()
                accuracyColor_cat = list()
                load_checkpoint(
                    'output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))
                print('doing model {0} for Table 1S'.format(modelNumber))
                clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
                clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
                clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
                clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))

                #the ratio of visual information encoded into memory
                shape_coeff = 1
                color_coeff = 1
                shape_coeff_half=.5
                color_coeff_half=.5
                shape_coeff_cat = 0
                color_coeff_cat = 0
                for i in range(perms):
                    # print('iterstart')
                    images, shapelabels = next(iter(test_loader_smaller))  # load up a set of digits
                    imgs = images.view(-1, 3 * 28 * 28).cuda()
                    colorlabels = torch.round(
                        imgs[:, 0] * 255)  
                    l1_act, l2_act, shape_act, color_act = activations(imgs)
                    shapepred, x, y, z = classifier_shapemap_test_imgs(shape_act, shapelabels, colorlabels, numItems,
                                                                       clf_shapeS, clf_shapeC)
                    colorpred, x, y, z = classifier_colormap_test_imgs(color_act, shapelabels, colorlabels, numItems,
                                                                       clf_colorC, clf_colorS)
                    
                    # one hot coding of labels before storing into the BP
                    shape_onehot = F.one_hot(shapepred, num_classes=20)
                    shape_onehot = shape_onehot.float().cuda()
                    color_onehot = F.one_hot(colorpred, num_classes=10)
                    color_onehot = color_onehot.float().cuda()
                    
                    #binding output when only maps are stored;  storeLabels=0
                    shape_out, color_out, L2_out, L1_out, shapelabel_junk, colorlabel_junk=BPTokens_with_labels(
                        bpsize, bpPortion, 0,shape_coeff, color_coeff, shape_act, color_act, l1_act, l2_act, shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    #this function is in mVAE
                    shapepredVisual, x, ssreportVisual, z = classifier_shapemap_test_imgs(shape_out, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpredVisual, x, ccreportVisual, z = classifier_colormap_test_imgs(color_out, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    #binding output that stores map activations + labels
                    shape_out_all, color_out_all, l2_out_all, l1_out_all, shape_label_out, color_label_out = BPTokens_with_labels(
                        bpsize, bpPortion, 1,shape_coeff, color_coeff, shape_act, color_act, l1_act, l2_act, shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)
                    shapepred, x, ssreport, z = classifier_shapemap_test_imgs(shape_out_all, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred, x, ccreport, z = classifier_colormap_test_imgs(color_out_all, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)
                    retrievedshapelabel = shape_label_out.argmax(1)
                    retrievedcolorlabel = color_label_out.argmax(1)
                    
                    # Compare accuracy against the original labels
                    accuracy_shape = torch.eq(shapelabels.cpu(), retrievedshapelabel.cpu()).sum().float() / numItems
                    accuracy_color = torch.eq(colorlabels.cpu(), retrievedcolorlabel.cpu()).sum().float() / numItems
                    accuracyShape.append(accuracy_shape)  # appends the perms
                    accuracyColor.append(accuracy_color)
                    accuracyShape_visual.append(ssreportVisual)
                    accuracyColor_visual.append(ccreportVisual)
                    accuracyShape_wlabels.append(ssreport)
                    accuracyColor_wlabels.append(ccreport)

                    # binding output that stores 50% of map activations + labels
                    shape_out_all_half, color_out_all_half, l2_out_all_half, l1_out_all_half, shape_label_out_half, color_label_out_half = BPTokens_with_labels(
                        bpsize, bpPortion, 1, shape_coeff_half, color_coeff_half, shape_act, color_act, l1_act, l2_act,
                        shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)

                    shapepred_half, x, ssreport_half, z = classifier_shapemap_test_imgs(shape_out_all_half, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred_half, x, ccreport_half, z = classifier_colormap_test_imgs(color_out_all_half, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)

                    retrievedshapelabel_half = shape_label_out_half.argmax(1)
                    retrievedcolorlabel_half = color_label_out_half.argmax(1)

                    accuracy_shape_half = torch.eq(shapelabels.cpu(), retrievedshapelabel_half.cpu()).sum().float() / numItems
                    accuracy_color_half = torch.eq(colorlabels.cpu(), retrievedcolorlabel_half.cpu()).sum().float() / numItems
                    accuracyShape_half.append(accuracy_shape_half)  # appends the perms
                    accuracyColor_half.append(accuracy_color_half)
                                        
                    # binding output that stores only labels with 0% visual information
                    shape_out_all_cat, color_out_all_cat, l2_out_all_cat, l1_out_all_cat, shape_label_out_cat, color_label_out_cat = BPTokens_with_labels(
                        bpsize, bpPortion, 1, shape_coeff_cat, color_coeff_cat, shape_act, color_act, l1_act, l2_act,
                        shape_onehot,
                        color_onehot,
                        numItems, 0, normalize_fact_familiar)
                    shapepred_cat, x, ssreport_cat, z = classifier_shapemap_test_imgs(shape_out_all_cat, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_shapeS, clf_shapeC)
                    colorpred_cat, x, ccreport_cat, z = classifier_colormap_test_imgs(color_out_all_cat, shapelabels, colorlabels,
                                                                              numItems,
                                                                              clf_colorC, clf_colorS)
                    retrievedshapelabel_cat = shape_label_out_cat.argmax(1)
                    retrievedcolorlabel_cat = color_label_out_cat.argmax(1)

                    accuracy_shape_cat = torch.eq(shapelabels.cpu(), retrievedshapelabel_cat.cpu()).sum().float() / numItems
                    accuracy_color_cat = torch.eq(colorlabels.cpu(), retrievedcolorlabel_cat.cpu()).sum().float() / numItems
                    accuracyShape_cat.append(accuracy_shape_cat)  # appends the perms
                    accuracyColor_cat.append(accuracy_color_cat)

                #append the accuracy for all models
                accuracyShapeModels.append(sum(accuracyShape) / perms)
                accuracyColorModels.append(sum(accuracyColor) / perms)
                accuracyShapeVisualModels.append(sum(accuracyShape_visual)/perms)
                accuracyColorVisualModels.append(sum(accuracyColor_visual)/perms)
                accuracyShapeWlabelsModels.append(sum(accuracyShape_wlabels) / perms)
                accuracyColorWlabelsModels.append(sum(accuracyColor_wlabels) / perms)
                accuracyShapeModels_half.append(sum(accuracyShape_half) / perms)
                accuracyColorModels_half.append(sum(accuracyColor_half) / perms)
                accuracyShapeModels_cat.append(sum(accuracyShape_cat) / perms)
                accuracyColorModels_cat.append(sum(accuracyColor_cat) / perms)
                  
            shape_dotplots_models.append(torch.stack(accuracyShapeModels).view(1,-1))
            totalAccuracyShape.append(torch.stack(accuracyShapeModels).mean())
            totalSEshape.append(torch.stack(accuracyShapeModels).std()/math.sqrt(numModels))
            
            color_dotplots_models.append(torch.stack(accuracyColorModels).view(1,-1))
            totalAccuracyColor.append(torch.stack(accuracyColorModels).mean())
            totalSEcolor.append(torch.stack(accuracyColorModels).std() / math.sqrt(numModels))
            
            shapeVisual_dotplots_models.append(torch.stack(accuracyShapeVisualModels).view(1,-1))
            totalAccuracyShape_visual.append(torch.stack(accuracyShapeVisualModels).mean())
            totalSEshapeVisual.append(torch.stack(accuracyShapeVisualModels).std()/math.sqrt(numModels))

            colorVisual_dotplots_models.append(torch.stack(accuracyColorVisualModels).view(1,-1))
            totalAccuracyColor_visual.append(torch.stack(accuracyColorVisualModels).mean())
            totalSEcolorVisual.append(torch.stack(accuracyColorVisualModels).std() / math.sqrt(numModels))
            
            shapeWlabels_dotplots_models.append(torch.stack(accuracyShapeWlabelsModels).view(1,-1))
            totalAccuracyShapeWlabels .append(torch.stack(accuracyShapeWlabelsModels).mean())
            totalSEshapeWlabels.append(torch.stack(accuracyShapeWlabelsModels).std() / math.sqrt(numModels))

            colorWlabels_dotplots_models.append(torch.stack(accuracyColorWlabelsModels).view(1,-1))
            totalAccuracyColorWlabels.append(torch.stack(accuracyColorWlabelsModels).mean())
            totalSEcolorWlabels.append(torch.stack(accuracyColorWlabelsModels).std() / math.sqrt(numModels))

            shape_half_dotplots_models.append(torch.stack(accuracyShapeModels_half).view(1,-1))
            totalAccuracyShape_half.append(torch.stack(accuracyShapeModels_half).mean())
            totalSEshape_half.append(torch.stack(accuracyShapeModels_half).std() / math.sqrt(numModels))

            color_half_dotplots_models.append(torch.stack(accuracyColorModels_half).view(1,-1))
            totalAccuracyColor_half.append(torch.stack(accuracyColorModels_half).mean())
            totalSEcolor_half.append(torch.stack(accuracyColorModels_half).std() / math.sqrt(numModels))
            
            shape_cat_dotplots_models.append(torch.stack(accuracyShapeModels_cat).view(1,-1))
            totalAccuracyShape_cat.append(torch.stack(accuracyShapeModels_cat).mean())
            totalSEshape_cat.append(torch.stack(accuracyShapeModels_cat).std() / math.sqrt(numModels))
            
            color_cat_dotplots_models.append(torch.stack(accuracyColorModels_cat).view(1,-1))
            totalAccuracyColor_cat.append(torch.stack(accuracyColorModels_cat).mean())
            totalSEcolor_cat.append(torch.stack(accuracyColorModels_cat).std() / math.sqrt(numModels))           

    print(shape_dotplots_models)
    print(color_dotplots_models)
    print(shapeVisual_dotplots_models)
    print(colorVisual_dotplots_models)
    print(shapeWlabels_dotplots_models)
    print(colorWlabels_dotplots_models) 
    print(shape_half_dotplots_models)
    print(color_half_dotplots_models)
    print(shape_cat_dotplots_models)
    print(color_cat_dotplots_models)
    
    #write the outputs
    outputFile.write('Table 1S, accuracy of ShapeLabel')
 
    #accuracy of retrieving lebals (stored: visual information +labels)
    for i in range(len(setSizes)):
        outputFile.write('\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape[i],totalSEshape[i] ))  
    outputFile.write('\n\nTable 3, accuracy of ColorLabel')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor[i], totalSEcolor[i]))

    #accuracy of retrieving visual (stored: visual infromation only)
    outputFile.write('\n\nTable 3, accuracy of shape map with no labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_visual[i], totalSEshapeVisual[i]))
    outputFile.write('\n\nTable 3, accuracy of color map with no labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_visual[i],
                                                                 totalSEcolorVisual[i]))
    #accuracy of retriveing visual (stored: visual + labels)
    outputFile.write('\n\nTable 3, accuracy of shape map with labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShapeWlabels[i],
                                                                totalSEshapeWlabels[i]))
    outputFile.write('\n\nTable 3, accuracy of color map with labels')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColorWlabels[i],
                                                                 totalSEcolorWlabels[i]))
        
    #accuracy of retriveing labels (stored: 50% visual + labels)
    outputFile.write('\n\nTable 3, accuracy of ShapeLabel for 50% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_half[i], totalSEshape_half[i]))
    outputFile.write('\n\nTable 3, accuracy of ColorLabel for 50% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_half[i], totalSEcolor_half[i]))
        
    #accuracy of retriveing labels (stored: only labels)  
    outputFile.write('\n\nTable 3, accuracy of ShapeLabel for 0% visual')    
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyShape_cat[i], totalSEshape_cat[i]))
    outputFile.write('\n\nTable 3, accuracy of ColorLabel for 0% visual')
    for i in range(len(setSizes)):
        outputFile.write(
            '\nSS {0} mean is {1:.4g} and SE is {2:.4g} '.format(setSizes[i], totalAccuracyColor_cat[i], totalSEcolor_cat[i]))
######This part is to detect whether a stimulus is novel or familiar#################
if noveltyDetectionFlag==1:
    perms=smallpermnum
  
    
    #accuracy of detecting a familiar vs novel item across different models and for "perms" number of binding pools (every time, a new BP with new sets of weights is generated)
    acc_fam=torch.zeros(numModels,perms)
    acc_nov=torch.zeros(numModels,perms)
    for modelNumber in range (1,numModels+1):   
        #this function is in tokens_capacity.py
        acc_fam[modelNumber-1,:], acc_nov[modelNumber-1,:]= novelty_detect( perms, bpsize, bpPortion, shape_coeff, color_coeff, normalize_fact_familiar,
                  normalize_fact_novel, modelNumber, test_loader_smaller)
        
    #mean accuracy of detecting whether the stimulus is novel or familiar
    mean_fam=acc_fam.view(1,-1).mean()
    fam_SE= acc_fam.view(1,-1).std()/(len(acc_fam.view(1,-1)))  
    mean_nov=acc_nov.view(1,-1).mean()
    nov_SE= acc_nov.view(1,-1).std()/(len(acc_nov.view(1,-1)))
    
    #write the outputs (mean and standard error) in the file
    outputFile.write(
            '\accuracy of detecting the familiar shapes : mean is {0:.4g} and SE is {1:.4g} '.format(mean_fam, fam_SE))    
    outputFile.write(
            '\naccuracy of detecting the novel shapes : mean is {0:.4g} and SE is {1:.4g} '.format(mean_nov, nov_SE))
outputFile.close()
