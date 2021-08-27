"""
Helper functions adapted from the Algonauts 2021 "development kit"
(https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/Algonauts2021_devkit)
"""

import os
import pickle
import numpy as np
from PIL import Image
from decord import VideoReader

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


def sample_video_frames(file, n_frames=16):
    """This function takes a mp4 video file as input and returns
    an array of uniformly sampled frames.

    Args:
        file (str): path to mp4 video file.
        n_frames (int): number of frames to select with uniform frame sampling.

    Returns:
        video_frames: list of frames as PIL images.
        num_frames (int): number of sampled frames.

    """

    video = VideoReader(file)
    total_frames = len(video)
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=np.int)

    video_frames = []
    for i in frame_indices:
        video_frames.append(Image.fromarray(video[i].asnumpy()))

    return video_frames, n_frames


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle.Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()

    return ret_di


def load_fmri(fmri_dir, sub, ROI):
    """This function loads and averages across sessions fMRI data for a given
    participant and region of interest (ROI).

    Args:
        fmri_dir (str): path to fMRI data.
        sub (str): participant number path.
        ROI (str): name of ROI.

    Returns:
        ROI_data_train (np array): matrix containing average fMRI responses to
            training videos (n_train_videos x n_voxels).

    """

    # Load ROI data
    ROI_file = os.path.join(fmri_dir, sub, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # average ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis=1)

    return ROI_data_train


def vectorised_correlation(x, y):
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)+1e-8
    y_std = y.std(axis=dim, keepdims=True)+1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()


def perform_encoding(train_features, fmri_dir, sub, ROI):
    """This function linearly regresses pca-reduced features/activations of a
    neural network to the fMRI responses of a given subject in a given region
    of interest (ROI). The fitted model is evaluated on a held-out, validation
    set.

    Args:
        train_features (np array): matrix containing pca features of a neural
            network to all training videos (n_train_videos x n_components).
        fmri_dir (str): path to fMRI data.
        sub (str): participant number path.
        ROI (str): name of ROI.

    Returns:
        voxelwise_corr (float): correlation averaged across voxels for given
            subject and ROI.

    """

    fmri_train = load_fmri(fmri_dir, sub, ROI)

    # create training and validation sets
    val_features = train_features[900:, :]
    train_features = train_features[:900, :]
    fmri_train = fmri_train[:900, :]
    fmri_val = fmri_train[900:, :]

    # perform multiple multivariate regression
    reg = MultiOutputRegressor(LinearRegression())
    reg.fit(train_features, fmri_train)
    pred_fmri = reg.predict(val_features)

    # correlation between predictions and ground truth for each voxel
    corr = vectorised_correlation(fmri_val, pred_fmri)
    # mean correlation across voxels
    voxelwise_corr = round(corr.mean(), 6)

    return voxelwise_corr
