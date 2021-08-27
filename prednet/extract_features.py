import cv2
import numpy as np
from utils.sample_video_frames import sample_video_frames

import torch
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


def extract_activations(model, video_list, layer, batch_size=1):
    """This function extracts the activations/features of a specific error
    layer of prednet to a set of videos. The activations are averaged across
    video frames.

        Args:
            model (tensorflow class): prednet model.
            video_list (list): list containing video paths.
            layer (int): prednet layer number.
            batch_size (int): default is 1 for one video.

        Returns:
            activations (np array): matrix storing model layer activations
                averaged across video frames (n_videos x n_layer_units).
    """

    error_units = 1

    activations = np.zeros((len(video_list), error_units))

    for i, video in enumerate(video_list):
        print(f"Processing video #{i}")

        video_frames, num_frames = sample_video_frames(video)

        # preprocess video frames
        video_frames = video_frames / 255.0
        video_frames = np.array([cv2.resize(video_frames[frame], (128, 160))
                                 for frame in range(num_frames)])
        video_frames = np.expand_dims(video_frames, axis=0)
        video_frames = np.transpose(video_frames, (0, 1, 4, 3, 2))

        # pass video frames through the model
        layer_error = model.predict(video_frames, batch_size)

        # average layer activations/errors across frames
        layer_error = layer_error.squeeze(axis=0)[:, layer-1]
        avg_activations = (np.sum(layer_error, axis=0) / float(num_frames)).flatten()

        activations[i] = avg_activations

    return activations


def apply_pca(train_activations):
    """This function applies principal component analysis to the training
    activations/features of a prednet model.

        Args:
            train_activations (np array): matrix storing model layer activations
                averaged across training video frames (n_videos x n_layer_units).

        Returns:
            train_features (np array): matrix with pca-reduced activations to
                every video (n_videos x n_pca_components).
    """

    n_components = 100
    train_activations = StandardScaler().fit_transform(train_activations)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(train_activations)
    train_features = pca.transform(train_activations)

    return train_features
