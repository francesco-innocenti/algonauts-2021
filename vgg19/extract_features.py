import os
import glob
import numpy as np
from utils import sample_video_frames

import torch
from torchvision import transforms as trn
from torch.autograd import Variable as V
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# seed for reproducibility
seed = 24
# torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# python RNG
np.random.seed(seed)


def extract_activations(model, video_list, save_dir, layer):
    """This function extracts the activations (features) of a given layer of
    a pytorch neural network to a set of videos and save them in a specified
    directory. The activations are averaged over the frames of every video.

    Parameters
    ----------
    model :
        pytorch model.
    video_list : list
        list containing path to all videos.
    save_dir : str
        save path for extracted activations.
    layer : str
        integer specifying layer number.
    """

    layer_index = [int(n) for n in layer.split('_') if n.isdigit()][0]-1

    # preprocessing function
    preprocess = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for i, video in enumerate(video_list):
        video_file_name = os.path.split(video)[-1].split(".")[0]
        video_frames, num_frames = sample_video_frames(video)

        activations = []
        for frame, image in enumerate(video_frames):
            input_image = V(preprocess(image).unsqueeze(0))
            if torch.cuda.is_available():
                input_image = input_image.cuda()
            layer_outputs = model.forward(input_image)
            if frame == 0:
                activations.append(layer_outputs[layer_index].ravel())
            else:
                activations[0] = activations[0] + layer_outputs[layer_index].ravel()

        # average layer activations across frames
        avg_layer_activations = np.array([activations]) / float(num_frames)

        # save activations for a particular video
        save_path = os.path.join(save_dir, video_file_name + "_" + layer + ".npy")
        np.save(save_path, avg_layer_activations)


def apply_pca(activations_dir, save_dir, layer):
    """This function preprocesses a neural network's features using PCA and
    save the results in a specified directory.

    Parameters
    ----------
    activations_dir : str
        save path for extracted activations.
    save_dir : str
        save path for extracted PCA features.
    """

    # number of PCA components
    n_components = 100

    # get path of activations of a layer to all videos
    activations_file_list = glob.glob(activations_dir + '/*.npy')
    # sort activations in ascending order
    activations_file_list.sort()
    # load one activation to later get dimension
    feature_dim = np.load(activations_file_list[0], allow_pickle=True)[0][0].shape[0]

    # initialise data matrix for PCA - #videos x layer dimension
    x = np.zeros((len(activations_file_list), feature_dim))

    # fill matrix with activations of every video (for a given layer)
    for i, video_activations in enumerate(activations_file_list):
        temp = np.load(video_activations, allow_pickle=True)[0][0].detach().numpy()
        x[i, :] = temp

    # training data
    x_train = x[:1000, :]

    # scale training and test sets
    x_train = StandardScaler().fit_transform(x_train)

    # find principal components of training set
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(x_train)

    # project data onto components - #videos x #components
    x_train = pca.transform(x_train)

    # save results
    train_save_path = os.path.join(save_dir, "pca_activations_" + layer)
    np.save(train_save_path, x_train)

