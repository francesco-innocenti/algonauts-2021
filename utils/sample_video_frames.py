import numpy as np
from PIL import Image
from decord import VideoReader


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
