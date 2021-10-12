# MLR
This is a model of working memory coupled with visual knowledge

-The visual knowledge represented by mVAE is trained on MNIST and f-MNIST
-The skip connection (from L1 to L5) was trained on cropped MNIST and f-MNIST that were presented in different locations with some degrees of rotation


To train the model:
Run the Training.py and save the model in output1 (you can run 10 times and save them in differnt directories from output1 to outut10)

To train the classifiers:

Run the Training_classifier.py. This trains and saves the classifiers trained on each model (output1 to output10)


To get the figures in the manuscript:

run the plot.py file. Within the file there are flags that can be set to 1 or zero corresponding to the figures/tables in the manuscript

To get the prre-trained model in addition to the simulation results please visit the OSF webpage (https://osf.io/tpzqk/). Run the plots.py file to reproduce the results

# requirements:
The model was programmed in Python 3.7.6 . in a torch environment version 1.3.1. The imported packages are listed here: torch, numpy , torch.nn , torch.nn.functional, torch.optim , imageio, os, copy, matplotlib.pyplot , matplotlib.image , torchvision,  datasets, transforms, utils, torch.autograd, Variable, torchvision.utils, save_image, sklearn, svm, sklearn.metrics, classification_report, confusion_matrix, tqdm, PIL, Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION, joblib, dump, load. 
