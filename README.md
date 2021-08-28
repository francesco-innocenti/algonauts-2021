# Algonauts 2021 Models
This is a repo evaluating models for predicting fMRI data associated with the 
Algonauts Project 2021 Challenge ([Cichy et al., 2021](https://arxiv.org/abs/2104.13714); 
[Cichy, Roig & Oliva, 2019](https://www.nature.com/articles/s42256-019-0127-z)). 
In brief, the challenge is to predict the fMRI responses of 10 participants to
over 1000 short natural videos. 

In this mini-project - which was conducted as part of 
[Neuromatch Academy 2021](https://academy.neuromatch.io) - we were interested 
in testing a biologically inspired model. We compared PredNet 
([Lotter, Kreiman & Cox, 2016](https://arxiv.org/abs/1605.08104), 
[2020](https://www.nature.com/articles/s42256-020-0170-9)), a network inspired 
by the influential neuroscience theory of predictive coding 
([Millidge, Seth & Buckley, 2021](https://arxiv.org/abs/2107.12979)), to one of 
the currently most predictive models of the visual system, VGG19 
([Simonyan & Zisserman, 2015](https://arxiv.org/abs/1409.1556)).

Here are some video predictions of a PredNet pretrained on a self-driving car 
dataset (see [Lotter, Kreiman & Cox, 2016](https://arxiv.org/abs/1605.08104)).

![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/actual_video_234.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/actual_video_390.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/actual_video_539.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/actual_video_587.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/actual_video_705.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/actual_video_976.gif)

![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/predicted_video_234.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/predicted_video_390.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/predicted_video_539.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/predicted_video_587.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/predicted_video_705.gif)
![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/prednet/video_predictions/predicted_video_976.gif)

We built two simple encoding models, extracting the activations of each network 
to the training videos (1000), reducing their dimensionality with PCA, and 
linearly regressing the components onto the fMRI responses of 9 visual regions 
in all subjects. We evaluated the fitted models on a held-out validation set of 
100 videos. Here are the results. 

![Alt Text](https://github.com/FrancescoInnocenti/Algonauts_2021_Models/blob/main/model_comparison.png)

Interestingly, we found that VGG19 - which has 3 orders of magnitude more 
parameters than PredNet - needs many more layers to match the performance of
PredNet.

### Code organisation

The code for the two models, PredNet and VGG19, is stored in different
directories (`prednet` and `vgg19`) because they have different dependencies.
Functions used for computation with both models are stored in a custom-made
mini-package called `algonauts`.