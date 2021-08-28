"""
This script predicts video frames with a pretrained PredNet.
"""

import glob
import cv2
import imageio
import numpy as np
from load_prednet import load_prednet
from algonauts.utils import sample_video_frames


# seed for reproducibility
seed = 24
np.random.seed(seed)

# model directory
model_dir = './model_data_keras2/'

# load pretrained model in video prediction mode
prednet = load_prednet(model_dir, output_mode='prediction')
batch_size = 1

# list of video paths
video_dir = '/AlgonautsVideos268_All_30fpsmax'
video_list = glob.glob(video_dir + '/*.mp4')
video_list.sort()
n_train_videos = 1000
n_predictions = 6

video_indices = np.random.choice(range(n_train_videos), n_predictions)

for i in video_indices:
    frames, num_frames = sample_video_frames(video_list[i])
    # preprocess
    frames = frames / 255.0
    frames = np.array([cv2.resize(frames[frame], (128, 160)) for frame in
                       range(num_frames)])
    frames = np.expand_dims(frames, axis=0)
    frames = np.transpose(frames, (0, 1, 4, 3, 2))

    # predict next video frame
    pred_frames = prednet.predict(frames, batch_size)

    # preprocess original frames
    frames = np.transpose(frames, axes=(0, 1, 4, 3, 2))
    frames = frames.squeeze(0)

    # preprocess predictions
    pred_frames = np.transpose(pred_frames, axes=(0, 1, 4, 3, 2))
    pred_frames = pred_frames.squeeze(0)

    fps = 5.33
    imageio.mimwrite(f"actual_video_{i}.gif", frames, fps)
    imageio.mimwrite(f"predicted_video_{i}.gif", pred_frames, fps)
