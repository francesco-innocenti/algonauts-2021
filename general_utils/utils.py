from decord import VideoReader
import numpy as np
from PIL import Image
import pickle
import os


def sample_video_frames(file, n_frames=16):
    """This function takes a mp4 video file as input and returns
    an array of uniformly sampled frames.

    Args
    ----------
    file : str
        path to mp4 video file
    n_frames : int
        number of frames to select with uniform frame sampling

    Returns
    -------
    frames: list of frames as PIL images
    num_frames: number of sampled frames

    """

    # read video file
    video = VideoReader(file)

    # get total number of video frames
    total_frames = len(video)

    # create frame indices
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=np.int)

    video_frames = []

    # list of video frames as PIL images
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
    """This function loads fMRI data into a numpy array for a given
    participant and ROI.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    sub : str
        participant number path
    ROI : str
        name of ROI.

    Returns
    ---------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI.
    """

    # Load ROI data
    ROI_file = os.path.join(fmri_dir, sub, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # average ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis=1)

    return ROI_data_train


def vectorized_correlation(x, y):
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)+1e-8
    y_std = y.std(axis=dim, keepdims=True)+1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()
