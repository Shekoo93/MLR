# MLR
This is A model of working memory coupled with visual knowledge

-The mVAE is trained on MNIST and f-MNIST
-THe skip connection (from L1 to L5) was trained on MNIST and f-MNIST that were presented in different locations with some degrees of rotation


To train the model:
Run the Training.py and save the model in output1 (you can run 10 times and save them in differnt directories from output1 to outut10)

To train the classifiers:

Run the Training_classifier.py. This saves the classifiers trained on each model (output1 to output10)


To get the figures in the manuscript:

1. run the plot.py file. Within the file there are flags that can be set to 1 or zero corresponding to the figures/tables in the manuscript
