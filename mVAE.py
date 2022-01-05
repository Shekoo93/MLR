
# MNIST VAE from http://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb 
# Modified by Brad Wyble and Shekoofeh Hedayati
# Modifications:
# Colorize transform that changes the colors of a grayscale image
# colors are chosen from 10 options:
colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "teal"]
# specified in "colorvals" variable below

# also there is a skip connection from the first layer to the last layer to enable reconstructions of new stimuli
# and the VAE bottleneck is split, having two different maps
# one is trained with a loss function for color only (eliminating all shape info, reserving only the brightest color)
# the other is trained with a loss function for shape only


# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os

from config import numcolors, args
from dataloader import notMNIST
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

global colorlabels

#generating a large vector of color labels
colorlabels = np.random.randint(0, 10, 1000000)
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [colorrange, .5, .5]
]

try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

# Enter the picture address
# Return tensor variable
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def Colorize_func(img): 
    global numcolors,colorlabels    # necessary because we will be modifying this counter variable

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index
    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')

    return img


#the secret function will be used for testing the classifiers (it has fix color labels)
def Colorize_func_secret(img,npflag = 0):
    global numcolors,colorlabels  
    
    thiscolor = colorlabels[numcolors]  
    thiscolor = np.random.randint(10)
    rgb = colorvals[thiscolor];  
    

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    
    np_img[0,0,0] = thiscolor   #secretely embed the color label inside
    img = Image.fromarray(np_img, 'RGB')
    if npflag ==1:
        img = backup

    return img



#to choose a specific class in case is necessary
def data_filter (data_type, selected_labels):
  data_trans= copy.deepcopy(data_type)
  data_type_labels= data_type.targets
  idx_selected= np.isin(data_type_labels, selected_labels)
  idx_selected=torch.tensor(idx_selected)
  data_trans.targets= data_type_labels[idx_selected]
  data_trans.data = data_type.data[idx_selected]
  return data_trans

#to extract the color labels
def thecolorlabels(datatype):    
    colornumstart = 0
    coloridx = range(colornumstart, len(datatype))
    labelscolor = colorlabels[coloridx]
    return torch.tensor(labelscolor)


bs = 100 #batch size
nw = 8 #number of workers

# MNIST and Fashion MNIST Datasets
train_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=True,
                               transform=transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)
test_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=False,
                              transform=transforms.Compose([Colorize_func, transforms.ToTensor()]), download=False)

ftrain_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=True,
                                       transform=transforms.Compose([Colorize_func, transforms.ToTensor()]),
                                       download=True)
ftest_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=False,
                                      transform=transforms.Compose([Colorize_func, transforms.ToTensor()]),
                                      download=False)

train_mnist_labels= train_dataset_MNIST.targets
ftrain_dataset.targets=ftrain_dataset.targets+ 10 #adding 10 to each f-mnist dataset label to make it distinguishable from mnist
train_fmnist_labels=ftrain_dataset.targets

test_mnist_labels= test_dataset_MNIST.targets
ftest_dataset.targets=ftest_dataset.targets+10
test_fmnist_label= ftest_dataset.targets

#skip connection dataset
train_skip_mnist= datasets.MNIST(root='./mnist_data/', train=True,
                               transform=transforms.Compose([Colorize_func,transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8), transforms.ToTensor()]), download=True)
train_skip_fmnist= datasets.FashionMNIST(root='./fashionmnist_data/', train=True,
                                       transform=transforms.Compose([Colorize_func, transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8),transforms.ToTensor()]),
                                       download=True)



train_dataset_skip= torch.utils.data.ConcatDataset((train_skip_mnist ,train_skip_fmnist)) #training skip connection with all images

train_dataset = torch.utils.data.ConcatDataset((train_dataset_MNIST ,ftrain_dataset))
test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST ,ftest_dataset))

train_loader_noSkip = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True,num_workers=nw)
train_loader_skip = torch.utils.data.DataLoader(dataset=train_dataset_skip, batch_size=bs, shuffle=True,  drop_last= True,num_workers=nw)
test_loader_noSkip = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True,num_workers=nw)
test_loader_skip = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,  drop_last=True,num_workers=nw)


#train and test the classifiers on MNIST and f-MNIST (the batch size that contains all the images)
bs_tr=120000
bs_te=20000

train_loader_class= torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs_tr, shuffle=True,num_workers=nw)
test_loader_class= torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_te, shuffle=False,num_workers=nw)


#the modified VAE
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        #encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        
        #decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
        #skip connection
        self.fc7 = nn.Linear(h_dim1, h_dim1)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        hskip = F.relu(self.fc7(h))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), hskip  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_noskip(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_color(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_shape(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_all(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = (F.relu(self.fc5(h)) + hskip)
        return torch.sigmoid(self.fc6(h))

    def decoder_skip(self, z_shape, z_color, hskip):
        return torch.sigmoid(self.fc6(hskip))


    def forward_layers(self, l1,l2, layernum,whichdecode):
        hskip = F.relu(self.fc7(l1))
        if layernum == 1:

           h = F.relu(self.fc2(l1))
           mu_shape = self.fc31(h)
           log_var_shape = self.fc32(h)
           mu_color = self.fc33(h)
           log_var_color = self.fc34(h)
           z_shape = self.sampling(mu_shape, log_var_shape)
           z_color = self.sampling(mu_color, log_var_color)
        elif layernum==2:
            
            h=l2
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'all'):
            output = self.decoder_all(z_shape, z_color, hskip)  #decodes via skip and noskip
        elif (whichdecode == 'skip'):
            output = self.decoder_skip(z_shape, z_color, hskip) #decodes only via skip
        else:
            output = self.decoder_noskip(z_shape, z_color, hskip) #decodes via noskip

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, whichdecode, detatchgrad='none'):
        mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x.view(-1, 784 * 3))
        if (detatchgrad == 'shape'):
            z_shape = self.sampling(mu_shape, log_var_shape).detach()
        else:
            z_shape = self.sampling(mu_shape, log_var_shape)

        if (detatchgrad == 'color'):
            z_color = self.sampling(mu_color, log_var_color).detach()
        else:
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'all'):
            output = self.decoder_all(z_shape, z_color, hskip)
        elif (whichdecode == 'noskip'):
            output = self.decoder_noskip(z_shape, z_color, 0)
        elif (whichdecode == 'skip'):
            output = self.decoder_skip(0, 0, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color, 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape, 0, 0)

        return output, mu_color, log_var_color, mu_shape, log_var_shape


# reload a saved file
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    vae.load_state_dict(checkpoint['state_dict'])
    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae.eval()
    return vae


# build model
vae = VAE(x_dim=784 * 3, h_dim1=256, h_dim2=128, z_dim=8)


if torch.cuda.is_available():
    vae.cuda()
    print('CUDA')

optimizer = optim.Adam(vae.parameters())


#return reconstruction error (this will be used to train the skip connection)
def loss_function(recon_x, x, mu, log_var, mu_c, log_var_c):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784 * 3), reduction='sum')
    return BCE 


#loss function for shape map
def loss_function_shape(recon_x, x, mu, log_var):
    # make grayscale reconstruction
    grayrecon = recon_x.view(bs, 3, 28, 28).mean(1)
   
    grayrecon = torch.stack([grayrecon, grayrecon, grayrecon], dim=1)
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(grayrecon.view(-1, 784 * 3), x.view(-1, 784 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD

#loss function for color map
def loss_function_color(recon_x, x, mu, log_var):
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(bs, 3, 784)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(recon[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(recon[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(recon[:, 2, :], -1, keepdim=True)

 #now build a new reconsutrction that has only the max color, and no shape information at all
 
    recon[:, 0, :] = maxr
    recon[:, 1, :] = maxg
    recon[:, 2, :] = maxb
    recon = recon.view(-1, 784 * 3)
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, 784 * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch, whichdecode):
    global numcolors
    colorlabels = np.random.randint(0, 10,1000000)  # regenerate the list of color labels at the start of each test epoch
    numcolors = 0
    vae.train()
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) #the latent space is trained on MNIST and f-MNIST
    dataiter_skip= iter(train_loader_skip) #The skip connection is trained on notMNIST
    count=0
    for i in range(1, len(train_loader_noSkip)):
        if (whichdecode == 'iterated'):
           loader=tqdm(train_loader_noSkip)
           data = dataiter_noSkip .next()
           data = data[0].cuda()
           count=count+1
           detachgrad = 'none'
           optimizer.zero_grad()
           # ADD COMMENTS
           if count% 3 == 0:
                    whichdecode_use = 'noskip'
                    detachgrad = 'color'

           elif count% 3==1:
                    whichdecode_use = 'noskip'
                    detachgrad = 'shape'

           else:
               data = dataiter_skip.next()
               data = data[0].cuda()
               whichdecode_use = 'skip'
               detachgrad = 'none'

        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use, detachgrad)
        if (whichdecode == 'iterated'):  # if yes, randomly alternate between using and ignoring the skip connections
            if count % 3 == 0:  # one of out 3 times, let's use the skip connection
                loss = loss_function_shape(recon_batch, data, mu_shape,
                                           log_var_shape)  # change the order, was color and shape
                loss.backward()

            elif count% 3 == 1:
                loss = loss_function_color(recon_batch, data, mu_color, log_var_color)
                loss.backward()

            else:
                loss = loss_function(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
                loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {loss.item():.5f};'
            )
        )
        sample_size = 25
        if count % 500== 0:
            vae.eval()
            sample = data[:sample_size]
            with torch.no_grad():
                reconskip, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'skip')
                reconb, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'noskip')
                reconc, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'color')
                recons, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'shape')
           
            utils.save_image(
                torch.cat([sample, reconskip.view(sample_size, 3, 28, 28), reconb.view(sample_size, 3, 28, 28),
                           reconc.view(sample_size, 3, 28, 28), recons.view(sample_size, 3, 28, 28)], 0),
                f'sample/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))


def test(whichdecode):
    vae.eval()
    global numcolors
    test_loss = 0
    testiter_noSkip = iter(test_loader_noSkip)  # the latent space is trained on MNIST and f-MNIST
    testiter_skip = iter(test_loader_skip)  # The skip connection is trained on notMNIST
    with torch.no_grad():
        for i in range(1, len(test_loader_noSkip)): # get the next batch


            data = testiter_noSkip.next()
            data = data[0].cuda()
            recon, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, 'noskip')

            # sum up batch loss
            test_loss += loss_function_shape(recon, data, mu_shape, log_var_shape).item()
            test_loss += loss_function_color(recon, data, mu_color, log_var_color).item()
            test_loss = loss_function(recon, data, mu_shape, log_var_shape, mu_color, log_var_color).item()

    print('Example reconstruction')
    data = data.cpu()
    data=data.view(bs, 3, 28, 28)
    save_image(data[0:8], f'{args.dir}/orig.png')
 

    print('Imagining a shape')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, 8).cuda() * 0
        zs = torch.randn(64, 8).cuda() * 1
        sample = vae.decoder_noskip(zs, zc, 0).cuda()
        sample=sample.view(64,3,28,28)
        save_image(sample[0:8], f'{args.dir}/sampleshape.png')


    print('Imagining a color')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, 8).cuda() * 1
        zs = torch.randn(64, 8).cuda() * 0
        sample = vae.decoder_noskip(zs, zc, 0).cuda()
        sample=sample.view(64, 3, 28, 28)
        save_image(sample[0:8], f'{args.dir}/samplecolor.png')


    test_loss /= len(test_loader_noSkip.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def activations(image):
    l1_act = F.relu(vae.fc1(image))
    l2_act = F.relu(vae.fc2(l1_act))
    mu_shape, log_var_shape, mu_color, log_var_color, hskip = vae.encoder(image)
    shape_act = vae.sampling(mu_shape, log_var_shape)
    color_act = vae.sampling(mu_color, log_var_color)
    return l1_act, l2_act, shape_act, color_act


def activation_fromBP(L1_activationBP, L2_activationBP, layernum):
    if layernum == 1:
        l2_act_bp = F.relu(vae.fc2(L1_activationBP))
        mu_shape = (vae.fc31(l2_act_bp))
        log_var_shape = (vae.fc32(l2_act_bp))
        mu_color = (vae.fc33(l2_act_bp))
        log_var_color = (vae.fc34(l2_act_bp))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    elif layernum == 2:
        mu_shape = (vae.fc31(L2_activationBP))
        log_var_shape = (vae.fc32(L2_activationBP))
        mu_color = (vae.fc33(L2_activationBP))
        log_var_color = (vae.fc34(L2_activationBP))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    return shape_act_bp, color_act_bp

#binding pool function for ONE ITEM (also BPTokens function can be used, but this code is faster for 1 item)
def BP(bp_outdim, l1_act, l2_act, shape_act, color_act, shape_coeff, color_coeff,l1_coeff,l2_coeff, normalize_fact):
    with torch.no_grad():
        bp_in1_dim = l1_act.shape[1]  # dim=256    #inputs to the binding pool
        bp_in2_dim = l2_act.shape[1]  # dim =128
        bp_in3_dim = shape_act.shape[1]  # dim=4
        bp_in4_dim = color_act.shape[1]  # dim=4
        
        #forward weigts from the mVAE layers to the BP
        c1_fw = torch.randn(bp_in1_dim, bp_outdim).cuda()     
        c2_fw = torch.randn(bp_in2_dim, bp_outdim).cuda()
        c3_fw = torch.randn(bp_in3_dim, bp_outdim).cuda()
        c4_fw = torch.randn(bp_in4_dim, bp_outdim).cuda()
        #backward weights from the BP to mVAE layers
        c1_bw = c1_fw.clone().t()
        c2_bw = c2_fw.clone().t()
        c3_bw = c3_fw.clone().t()
        c4_bw = c4_fw.clone().t()    
        BP_in_all = list()
        shape_out_BP_all = list()
        color_out_BP_all = list()
        BP_layerI_out_all = list()
        BP_layer2_out_all = list()

        for idx in range(l1_act.shape[0]):
            BP_in_eachimg = torch.mm(shape_act[idx, :].view(1, -1), c3_fw) * shape_coeff + torch.mm(
                color_act[idx, :].view(1, -1), c4_fw) * color_coeff  # binding pool inputs (forward activations)
            BP_L1_each = torch.mm(l1_act[idx, :].view(1, -1), c1_fw) * l1_coeff
            BP_L2_each = torch.mm(l2_act[idx, :].view(1, -1), c2_fw) * l2_coeff


            shape_out_eachimg = torch.mm(BP_in_eachimg , c3_bw)  # backward projections from BP to the vae
            color_out_eachimg = torch.mm(BP_in_eachimg , c4_bw)
            L1_out_eachimg = torch.mm(BP_L1_each , c1_bw)
            L2_out_eachimg = torch.mm(BP_L2_each , c2_bw)

            BP_in_all.append(BP_in_eachimg)  # appending and stacking images

            shape_out_BP_all.append(shape_out_eachimg)
            color_out_BP_all.append(color_out_eachimg)
            BP_layerI_out_all.append(L1_out_eachimg)
            BP_layer2_out_all.append(L2_out_eachimg)

            BP_in = torch.stack(BP_in_all)

            shape_out_BP = torch.stack(shape_out_BP_all)
            color_out_BP = torch.stack(color_out_BP_all)
            BP_layerI_out = torch.stack(BP_layerI_out_all)
            BP_layer2_out = torch.stack(BP_layer2_out_all)

            shape_out_BP = shape_out_BP / bp_outdim
            color_out_BP = color_out_BP / bp_outdim
            BP_layerI_out = (BP_layerI_out / bp_outdim ) * normalize_fact
            BP_layer2_out = BP_layer2_out / bp_outdim


        return BP_L1_each, shape_out_BP, color_out_BP, BP_layerI_out, BP_layer2_out


#binding pool that stores images along with labels. In this implementation we are only encoding one layer at a time (L1, L2, shape/color maps)
def BPTokens_with_labels(bp_outdim, bpPortion,storeLabels, shape_coef, color_coef, shape_act, color_act,l1_act,l2_act,oneHotShape, oneHotcolor, bs_testing, layernum, normalize_fact ):
    # Store and retrieve multiple items including labels in the binding pool 
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items  
    # storeLabels: 0 if labels are not stored, 1 if labels are stored
    # if labels are not available for the given items, matrices with zero elements are entered as the function's input
  
    with torch.no_grad():  # <---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all = list()  # will be used to accumulate the specific token linkages
        BP_in_all = list()  # will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]
        bp_in_L2_dim = l2_act.shape[1]
        oneHotShape = oneHotShape.cuda()

        oneHotcolor = oneHotcolor.cuda()
        bp_in_Slabels_dim = oneHotShape.shape[1]  # dim =20
        bp_in_Clabels_dim= oneHotcolor.shape[1]

        # will be used to accumulate the reconstructions
        shape_out_all = torch.zeros(bs_testing,bp_in_shape_dim).cuda()  
        color_out_all = torch.zeros(bs_testing,bp_in_color_dim).cuda()  
        L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
        L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
        shape_label_out=torch.zeros(bs_testing, bp_in_Slabels_dim).cuda()
        color_label_out = torch.zeros(bs_testing, bp_in_Clabels_dim).cuda()
        
        # make the randomized fixed weights to the binding pool
        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()
        L2_fw = torch.randn(bp_in_L2_dim, bp_outdim).cuda()
        shape_label_fw=torch.randn(bp_in_Slabels_dim, bp_outdim).cuda()
        color_label_fw = torch.randn(bp_in_Clabels_dim, bp_outdim).cuda()

        # ENCODING!  Store each item in the binding pool
        for items in range(bs_testing):  # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

            if layernum == 1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            elif layernum==2:
                BP_in_eachimg = torch.mm(l2_act[items, :].view(1, -1), L2_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coef + torch.mm(color_act[items, :].view(1, -1), color_fw) * color_coef  # binding pool inputs (forward activations)
                BP_in_Slabels_eachimg=torch.mm(oneHotShape [items, :].view(1, -1), shape_label_fw)
                BP_in_Clabels_eachimg = torch.mm(oneHotcolor[items, :].view(1, -1), color_label_fw)
                BP_in_Slabels_eachimg[:, notLink] = 0
                BP_in_Clabels_eachimg[:, notLink] = 0


            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            
            if storeLabels==1:
                BP_in_all.append(
                    BP_in_eachimg + BP_in_Slabels_eachimg + BP_in_Clabels_eachimg)  # appending and stacking images
                notLink_all.append(notLink)

            else:
                BP_in_all.append(BP_in_eachimg )  # appending and stacking images
                notLink_all.append(notLink)



        # now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items, 1)
        BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # divide by the token percent, as a normalizing factor

        BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
        notLink_all = torch.stack(notLink_all)  # this is the set of 0'd connections for each of the tokens

        # NOW REMEMBER
        for items in range(bs_testing):  # for each item to be retrieved
            BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
            if layernum == 1:
                L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
                L1_out_all[items,:] = (L1_out_eachimg / bpPortion ) * normalize_fact # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
            if layernum==2:

                L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
                L2_out_all[items, :] = L2_out_eachimg / bpPortion  #
            else:
                shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),shape_fw.t()).cuda()  # do the actual reconstruction
                color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
                shapelabel_out_each=torch.mm(BP_in_items[items, :].view(1, -1),shape_label_fw.t()).cuda()
                colorlabel_out_each = torch.mm(BP_in_items[items, :].view(1, -1), color_label_fw.t()).cuda()

                shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
                color_out_all[items, :] = color_out_eachimg / bpPortion
                shape_label_out[items,:]=shapelabel_out_each/bpPortion
                color_label_out[items,:]=colorlabel_out_each/bpPortion

    return shape_out_all, color_out_all, L2_out_all, L1_out_all,shape_label_out,color_label_out

#Store multiple items in the binding pool, then try to retrieve the token of item #1 using its shape as a cue
def BPTokens_binding_all(bp_outdim,  bpPortion, shape_coef,color_coef,shape_act,color_act,l1_act,bs_testing,layernum, shape_act_grey, color_act_grey):
    
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items  
    #layernum= either 1 (reconstructions from l1) or 0 (recons from the bottleneck
    with torch.no_grad(): #<---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all=list()  #will be used to accumulate the specific token linkages
        BP_in_all=list()    #will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottlenecks
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]  # neurons in the Bottleneck
        tokenactivation = torch.zeros(bs_testing)  # used for finding max token
        shape_out = torch.zeros(bs_testing,
                                    bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out= torch.zeros(bs_testing,
                                    bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        l1_out= torch.zeros(bs_testing, bp_in_L1_dim).cuda()


        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  #make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()

        #ENCODING!  Store each item in the binding pool
        for items in range (bs_testing):   # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  #list of 0'd BPs for this token
            if layernum==1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw)+torch.mm(color_act[items, :].view(1, -1), color_fw) # binding pool inputs (forward activations)

            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_all.append(BP_in_eachimg)  # appending and stacking images
            notLink_all.append(notLink)

        #now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items,1)
        BP_in_items = torch.sum(BP_in_items,0).view(1,-1)   #divide by the token percent, as a normalizing factor

        notLink_all=torch.stack(notLink_all)   # this is the set of 0'd connections for each of the tokens

        retrieve_item = 0
        if layernum==1:
            BP_reactivate = torch.mm(l1_act[retrieve_item, :].view(1, -1), L1_fw)
        else:
            BP_reactivate = torch.mm(shape_act_grey[retrieve_item, :].view(1, -1),shape_fw)  # binding pool retreival
        
        # Multiply the cued version of the BP activity by the stored representations
        BP_reactivate = BP_reactivate  * BP_in_items

        for tokens in range(bs_testing):  # for each token
            BP_reactivate_tok = BP_reactivate.clone()
            BP_reactivate_tok[0,notLink_all[tokens, :]] = 0  # set the BPs to zero for this token retrieval
            # for this demonstration we're assuming that all BP-> token weights are equal to one, so we can just sum the
            # remaining binding pool neurons to get the token activation
            tokenactivation[tokens] = BP_reactivate_tok.sum()

        max, maxtoken =torch.max(tokenactivation,0)   #which token has the most activation

        BP_in_items[0, notLink_all[maxtoken, :]] = 0  #now reconstruct color from that one token
        if layernum==1:

            l1_out = torch.mm(BP_in_items.view(1, -1), L1_fw.t()).cuda() / bpPortion  # do the actual reconstruction
        else:

            shape_out = torch.mm(BP_in_items.view(1, -1), shape_fw.t()).cuda() / bpPortion  # do the actual reconstruction of the BP
            color_out = torch.mm(BP_in_items.view(1, -1), color_fw.t()).cuda() / bpPortion

    return tokenactivation, maxtoken, shape_out,color_out, l1_out 


# defining the classifiers  
clf_ss = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for shape
clf_sc = svm.SVC(C=10, gamma='scale', kernel='rbf')  #classify shape map against color labels
clf_cc = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for color
clf_cs = svm.SVC(C=10, gamma='scale', kernel='rbf')#classify color map against shape labels


# training the shape map on shape labels and color labels 
def classifier_shape_train(whichdecode_use):
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    train_colorlabels = 0
    numcolors = 0
    vae.eval()
    with torch.no_grad():
            data,train_shapelabels  =next(iter(train_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            train_colorlabels = thecolorlabels(train_dataset)
            print('training shape bottleneck against color labels sc')
            clf_sc.fit(z_shape.cpu().numpy(), train_colorlabels)

            print('training shape bottleneck against shape labels ss')
            clf_ss.fit(z_shape.cpu().numpy(), train_shapelabels)

#testing the shape classifier (one image at a time)
def classifier_shape_test(whichdecode_use, clf_ss, clf_sc,verbose =0):
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    numcolors = 0
    test_colorlabels=0
    with torch.no_grad():
            data, test_shapelabels= next(iter (test_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            test_colorlabels = thecolorlabels(test_dataset)
            pred_ss = torch.tensor(clf_ss.predict(z_shape.cpu()))
            pred_sc = torch.tensor(clf_sc.predict(z_shape.cpu()))

            SSreport = torch.eq(test_shapelabels.cpu(), pred_ss).sum().float() / len(pred_ss)
            SCreport = torch.eq(test_colorlabels.cpu(), pred_sc).sum().float() / len(pred_sc)


            if verbose ==1:
                print('----*************---------shape classification from shape map')
                print(confusion_matrix(test_shapelabels, pred_ss))
                print(classification_report(test_shapelabels, pred_ss))
                print('----************----------color classification from shape map')
                print(confusion_matrix(test_colorlabels, pred_sc))
                print(classification_report(test_colorlabels, pred_sc))
    return pred_ss, pred_sc, SSreport, SCreport

#training the color map on shape and color labels
def classifier_color_train(whichdecode_use):
    vae.eval()
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    numcolors = 0
    train_colorlabels = 0
    with torch.no_grad():
            data, train_shapelabels = next(iter (train_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            train_colorlabels = thecolorlabels(train_dataset)
            print('training color bottleneck against color labels cc')
            clf_cc.fit(z_color.cpu().numpy(), train_colorlabels)

            print('training color bottleneck against shape labels cs')
            clf_cs.fit(z_color.cpu().numpy(), train_shapelabels)

# testing the color classifier (one image at a time)
def classifier_color_test(whichdecode_use, clf_cc, clf_cs,verbose=0):
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    numcolors = 0

    test_colorlabels = 0
    with torch.no_grad():
            data, test_shapelabels = next(iter(test_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
       
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            test_colorlabels = thecolorlabels(test_dataset)
            pred_cc = torch.tensor(clf_cc.predict(z_color.cpu()))
            pred_cs = torch.tensor(clf_cs.predict(z_color.cpu()))

            CCreport = torch.eq(test_colorlabels.cpu(), pred_cc).sum().float() / len(pred_cc)
            CSreport = torch.eq(test_shapelabels.cpu(), pred_cs).sum().float() / len(pred_cs)

            if verbose==1:
                print('----**********-------color classification from color map')
                print(confusion_matrix(test_colorlabels, pred_cc))
                print(classification_report(test_colorlabels, pred_cc))


                print('----**********------shape classification from color map')
                print(confusion_matrix(test_shapelabels, pred_cs))
                print(classification_report(test_shapelabels, pred_cs))

    return pred_cc, pred_cs, CCreport, CSreport



# testing on shape for multiple images stored in memory

def classifier_shapemap_test_imgs(shape, shapelabels, colorlabels,numImg, clf_shapeS, clf_shapeC,verbose = 0):

    global numcolors
 
    numImg = int(numImg)

    with torch.no_grad():
        predicted_labels=torch.zeros(1,numImg)
        shape = torch.squeeze(shape, dim=1)
        shape = shape.cuda()
        test_colorlabels = thecolorlabels(test_dataset)
        pred_ssimg = torch.tensor(clf_shapeS.predict(shape.cpu()))
     
        pred_scimg = torch.tensor(clf_shapeC.predict(shape.cpu()))

        SSreport = torch.eq(shapelabels.cpu(), pred_ssimg).sum().float() / len(pred_ssimg)
        SCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_scimg).sum().float() / len(pred_scimg)

        if verbose==1:
            print('----*************---------shape classification from shape map')
            print(confusion_matrix(shapelabels[0:numImg], pred_ssimg))
            print(classification_report(shapelabels[0:numImg], pred_ssimg))
            print('----************----------color classification from shape map')
            print(confusion_matrix(colorlabels[0:numImg], pred_scimg))
            print(classification_report(test_colorlabels[0:numImg], pred_scimg))
    return pred_ssimg, pred_scimg, SSreport, SCreport


#testing on color for multiple images stored in memory
def classifier_colormap_test_imgs(color, shapelabels, colorlabels,numImg, clf_colorC, clf_colorS,verbose = 0):
    
    numImg = int(numImg)
    with torch.no_grad():
      
        color = torch.squeeze(color, dim=1)
        color = color.cuda()
        test_colorlabels = thecolorlabels(test_dataset)

        pred_ccimg = torch.tensor(clf_colorC.predict(color.cpu()))
        pred_csimg = torch.tensor(clf_colorS.predict(color.cpu()))

        CCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_ccimg).sum().float() / len(pred_ccimg)
        CSreport = torch.eq(shapelabels.cpu(), pred_csimg).sum().float() / len(pred_csimg)

        if verbose == 1:
            print('----*************---------color classification from color map')
            print(confusion_matrix(test_colorlabels[0:numImg], pred_ccimg))
            print(classification_report(colorlabels[0:numImg], pred_ccimg))
            print('----************----------shape classification from color map')
            print(confusion_matrix(shapelabels[0:numImg], pred_csimg))
            print(classification_report(shapelabels[0:numImg], pred_csimg))

        return pred_ccimg, pred_csimg, CCreport, CSreport




