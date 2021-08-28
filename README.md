# Algonauts 2021 Models
This is a repo evaluating models for predicting fMRI data associated with the 
Algonauts Project 2021 Challenge (see [Cichy et al., 2021](https://arxiv.org/abs/2104.13714); 
[Cichy, Roig & Oliva, 2019](https://www.nature.com/articles/s42256-019-0127-z)). 
In brief, the challenge is to predict the fMRI responses of 10 participants to
over 1000 short natural videos. 

In this project - which was conducted as part of [Neuromatch Academy 2021 summer 
courses](https://academy.neuromatch.io) - we were interested in testing a 
biologically inspired model. We compared PredNet, a network inspired by the 
influential neuroscience theory of predictive coding ([Millidge, Seth & Buckley, 
2021](https://arxiv.org/abs/2107.12979)), to one of the currently most 
predictive models of the visual system, VGG19 
([Simonyan & Zisserman, 2015](https://arxiv.org/abs/1409.1556)).

## Code organisation

The code for the two models, PredNet and VGG19, is stored in different
directories (`prednet` and `vgg19`) because they have different dependencies.
Functions used for computation with both models are stored in a mini-package 
called `algonauts`.