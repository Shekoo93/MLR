

colornames = ["red", "blue","green","purple","yellow","cyan","orange","brown","pink","teal"]
# generic prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
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
import random
config.init()
from config import numcolors
global numcolors
global colorlabels
from PIL import Image
from mVAE import *
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

 #This function presents two colored digits to the model and binds them to two different tokens
 #then it tries to retrieve one token based on a shape cue, reporting the accuracy
def binding_cue(bs_testing, perms,bpsize, bpPortion, shape_coeff, color_coeff,samediff, modelNumber ):
    load_checkpoint('output{modelNumber}/checkpoint_threeloss_singlegrad200.pth'.format(modelNumber=modelNumber))              
    clf_shapeS = load('output{num}/ss{num}.joblib'.format(num=modelNumber))
    clf_shapeC = load('output{num}/sc{num}.joblib'.format(num=modelNumber))
    clf_colorC = load('output{num}/cc{num}.joblib'.format(num=modelNumber))
    clf_colorS = load('output{num}/cs{num}.joblib'.format(num=modelNumber))

    #if samediff parameter is "diff" then it will use two different digits, otherwise they will be the same
    MNISTwithindex = dataset_with_indices(datasets.MNIST)
    test_dataset_MNIST_NC = MNISTwithindex(root='./mnist_data/', train=False,
                                           transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]), download=False)
    tokenactivation = torch.zeros(bs_testing, perms)
    whichtoken = torch.zeros(perms)
    accuracy_shape= torch.zeros(perms)
    accuracy_color= torch.zeros(perms)
    test_filtered = data_filter(test_dataset_MNIST_NC, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])   #this filtering is redundant but makes it easy to check individual digits
    test_loader_smaller = torch.utils.data.DataLoader(dataset=test_filtered, batch_size=1, shuffle=True, num_workers=0)
                                                     
    #make 10 smaller subsets for each individual digit
    test_loader = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for digit in range(0,10):
        test_filtered = data_filter(test_dataset_MNIST_NC, digit)
        test_loader[digit] = torch.utils.data.DataLoader(dataset=test_filtered, batch_size=1, shuffle=True,num_workers=0)                                                  
    pil2tensor = transforms.ToTensor()  #used later
    pred_ss_all=torch.zeros(perms)
    pred_cc_all=torch.zeros(perms)
    color_labels_all=torch.zeros(perms)
    shape_labels_all=torch.zeros(perms)
    for rep in range(perms):
            image, shapelabel,idx = next(iter(test_loader_smaller))  # load up one digit
            colorlabel = torch.round(image[0,0,0,0] * 255)

            shapelabel2 = shapelabel
            if(samediff == 'diff'):     #if we should be using different digits, grab random labels to get a different one, otherwise we'll be using the same digit
                while(shapelabel2==shapelabel):
                    shapelabel2 = random.randint(0, 9)
            image2, shapelabel2,idx2 = next(iter(test_loader[shapelabel2]))  # load up second digit
            images_grey = torch.stack([image[:,0,:,:], image2[:,0,:,:]])  #make a copy to keep as grayscale
            images_grey = torch.stack([images_grey, images_grey, images_grey], dim=1)  # add 3 depth
            image = torch.from_numpy(np.array(image))
            image2 = torch.from_numpy(np.array(image2))
            images = torch.stack([image,image2])  #colorized copy       
            imgs = images.view(-1, 3 * 28 * 28).cuda()
            imgs_grey = images_grey.view(-1, 3 * 28 * 28).cuda()

            # generating encoder and latent activations
            l1_act, l2_act, shape_act, color_act = activations(imgs.float())
            l1_act_grey, l2_act_grey, shape_act_grey, color_act_grey = activations(imgs_grey)

            #now do the memory retrieval
            tokenactivation[:, rep], whichtoken[rep], shape_out, color_out, l1_out = BPTokens_binding_all(bpsize,
                                                                                                          bpPortion,shape_coeff,color_coeff,shape_act,color_act,l1_act,bs_testing, 0,shape_act_grey, color_act_grey)
                                                                                                         
                                                                                                                                                                                                                
            #predicting the retrived shape and color
            pred_ss= torch.tensor(clf_shapeS.predict(shape_out.cpu()))
            pred_cc = torch.tensor(clf_colorC.predict(color_out.cpu()))
            color_labels_all[rep]=colorlabel
            shape_labels_all[rep]=shapelabel            
            pred_ss_all[rep]=pred_ss
            pred_cc_all[rep]=pred_cc           
    accuracy_correctToken = (1 - whichtoken.sum() / perms)  # 0's are correct and 1's are incorrect
    
    #accuracy of finding the correct color/shape
    accuracy_color=sum(pred_cc_all==color_labels_all).item()/perms       
    accuracy_shape=sum(pred_ss_all==shape_labels_all).item()/perms
    return  accuracy_correctToken, accuracy_color, accuracy_shape

################################cross correlations for familiar vs. novel
 # This function stores and retrieves a given set size of items as either familiar (using bottleneck)_or novel items (using L1)
 # performance is evaluated with a pixelwise correlation between the input and output layers
 # it shows that performance is lower for the novels, and also drops off more quickly as set size increases
 # Moreover, if you try to encode the novels using the bottleneck, correlations are much worse.

def storeretrieve_crosscorrelation_test(setSize, perms, bpsize, bpPortion, shape_coeff, color_coeff, normalize_fact_familiar,
                  normalize_fact_novel, modelNumber, test_loader_smaller,famnovel,layernum,memory):
   
    #famnovel = "fam" = familiar items
    #layernum:  1 = layer 1, otherwise bottleneck (shape and color combined)
    #memory: determines whether the image is stored or not; memory=1: store the item, memory=0: do not store the item     
    corr_multiple = list()
    l1_coeff = 1
    l2_coeff =1
    tokenactivation = torch.zeros(setSize, perms)
    whichtoken = torch.zeros(perms)
    trans2 = transforms.ToTensor()
    for rep in range(perms):
        if (rep % 100 == 0):
            print('Fig 15 Perm #{num} out of {num2} for set size {num3}'.format(num=rep, num2=perms, num3=setSize))
        if famnovel =="fam" :
            images, shapelabels = next(iter(test_loader_smaller))  # load up a set of digits
        else:
            whichImg = np.random.randint(1, 7, setSize)
            transformations = transforms.Compose(
                [transforms.RandomRotation(10), transforms.RandomCrop(size=28, padding=8)])
            images = list()
            for img_novel in range(setSize):
                img = Image.open('{each}_thick.png'.format(each=whichImg[img_novel]))
                img = np.mean(img, axis=2)
                img[img < 64] = 0
                img_col = Colorize_func(img)
                image = transformations(img_col)
                image = trans2(image) * 1.3
                images.append(image)
            images = torch.stack(images)
        imgs = images.view(-1, 3 * 28 * 28).cuda()
        
        # generating encoder and latent activations
        l1_act, l2_act, shape_act, color_act = activations(imgs)
        
        #staright recon from L1 using skip
        if layernum==1:
            l1_act_tr = l1_act.clone()
            if memory==0:
                retrievals, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_act_tr, l2_act, 1, 'skip')
                                                                                                 
            else:
                l1_act_tr[l1_act != 0] = l1_act_tr[l1_act != 0] + 2
                l1_act_tr[l1_act_tr == 0] = -3

                shape_out_all, color_out_all, l2_out_all, l1_out_all = BPTokens(bpsize, bpPortion, shape_coeff,
                                                                                color_coeff,
                                                                                l1_coeff, l2_coeff,
                                                                                shape_act, color_act, l1_act_tr, l2_act,
                                                                                setSize, 1, normalize_fact_novel)             
                l1_out_all[l1_out_all < 0] = 0

                retrievals, mu_color, log_var_color, mu_shape, log_var_shape = vae.forward_layers(l1_out_all, l2_act,  1,'skip')                                                                                                                                                                                              
        else:
            if memory==0:
                retrievals = vae.decoder_noskip(shape_act, color_act, 0).cuda()
            else:
                shape_out_all, color_out_all, l2_out_all, l1_out_all = BPTokens(bpsize, bpPortion, shape_coeff,
                                                                                color_coeff,
                                                                                0, 0,
                                                                                shape_act, color_act, l1_act, l2_act,
                                                                                setSize, 0, normalize_fact_familiar)
                retrievals = vae.decoder_noskip(shape_out_all, color_out_all, 0).cuda()
                
        #Now do the correlations
        corr_fam = list()
        for n in range(imgs.shape[0]):
            a = imgs[n, :].cpu().numpy()
            b = retrievals[n, :].cpu().numpy()
            corr = np.corrcoef(a, b)
            corr_fam.append(corr[0, 1])
        corr_multiple.append(corr_fam)    
    return np.array(corr_multiple)

###################################### detecting whether a stimulus novel or familiar
#this function detects whether a given item is familiar or novel  comparing the cross-correlation
#between the original image and its reconstruction form the BN

def novelty_detect( perms, bpsize, bpPortion, shape_coeff, color_coeff, normalize_fact_familiar,
                  normalize_fact_novel, modelNumber, test_loader_smaller):
 
    trans2 = transforms.ToTensor()
    setSize=1   #detects one image at a time
    l1_coeff=l2_coeff=1
    acc_nov=torch.zeros(perms)
    acc_fam=torch.zeros(perms)
    for rep in range(perms):
        images_fam, shapelabels = next(iter(test_loader_smaller))  # load up a set of digits
        
        #create the novel dataset
        images_novel=[]
        for novel_imgs in range(len(images_fam)):
            whichImg = np.random.randint(1, 7)
            transformations = transforms.Compose(
                [transforms.RandomRotation(10), transforms.RandomCrop(size=28, padding=8)])
            img = Image.open('{each}_thick.png'.format(each=whichImg))
            img = np.mean(img, axis=2)
            img[img < 64] = 0
            img_col = Colorize_func(img)
            image = transformations(img_col)
            image = trans2(image) * 1.3
            images_novel.append(image)   
        images_novel=torch.stack(images_novel)
        images_all=torch.cat([images_fam, images_novel]) #combine novel images with familiar ones
        indices=torch.randperm(len(images_all))
        images_all=images_all[indices]  #randomize the images
        
        #compute the cross-correlation for images reconstructed from the bottleneck   
        imgs = images_all.view(-1, 3 * 28 * 28).cuda()        
        l1_act, l2_act, shape_act, color_act = activations(imgs)   
        retrievals = vae.decoder_noskip(shape_act, color_act, 0).cuda()
        corr= list()
        for n in range(imgs.shape[0]):
            a = imgs[n, :].cpu().numpy()
            b = retrievals[n, :].cpu().numpy()
            corr_each = np.corrcoef(a, b)
            corr.append(corr_each[0, 1])
               
        #ground truth vs. estimated novel shapes
        idx_nov_actual_np=np.where(indices >= len(images_fam)) #actual indices of novel shapes
        idx_nov_np=np.where(corr < np.array(.5)) #estimated indices of novel shapes
        idx_nov_actual=torch.tensor(idx_nov_actual_np).clone().detach()
        idx_nov=torch.tensor(idx_nov_np).clone().detach()
        
        #ground truth vs. estimated familiar shapes
        idx_fam_actual_np=np.where(indices < len(images_novel)) #actual indices of familair shapes
        idx_fam_np=np.where(corr >= np.array(.5)) #estimated indices of familiar shapes
        idx_fam_actual=torch.tensor(idx_fam_actual_np).clone().detach()
        idx_fam=torch.tensor(idx_fam_np).clone().detach()
        
        #compute the accuracy of selecting the novel infromation        
        acc_nov_br=idx_nov.view(1,-1).eq(idx_nov_actual.view(-1,1)).sum(0)
        acc_nov[rep]=sum(acc_nov_br).item()/len(images_novel)
        
        #compute the accuracy of selecting the familiar infromation
        acc_fam_br=idx_fam.view(1,-1).eq(idx_fam_actual.view(-1,1)).sum(0)
        acc_fam[rep]=sum(acc_fam_br).item()/len(images_fam)   
           
    return acc_fam, acc_nov
  










