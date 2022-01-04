# MLR

-MNIST VAE was adopted from http://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb 

This is a model of working memory coupled with visual knowledge

-The visual knowledge represented by mVAE is trained on MNIST and f-MNIST
-The skip connection (from L1 to L5) was trained on cropped MNIST and f-MNIST that were presented in different locations with some degrees of rotation


To train the model:
Run the Training.py and save the model in output1 (you can run 10 times and save them in different directories from output1 to output10)

To train the classifiers:

Run the Training_classifier.py. This trains and saves the classifiers trained on each model (output1 to output10)


To get the figures in the manuscript:

run the plot.py file. Within the file there are flags that can be set to 1 or zero corresponding to the figures/tables in the manuscript

To get the pre-trained model in addition to the simulation results please visit the OSF webpage (https://osf.io/tpzqk/). Run the plots.py file to reproduce the results

Package Requirements are as listed in requirements.txt

-tokens_capacity.py file should not be changed. It consists of functions that compute cross correlations for novel vs. familiar shapes , detecting whether a given stimulus is novel or familiar and binding test
