"""
This script predicts video frames with a pretrained PredNet.
"""

import glob
import cv2
import imageio
import os
import numpy as np
from load_prednet import load_prednet
from algonauts.utils import sample_video_frames


# seed for reproducibility
seed = 24
np.random.seed(seed)

# load pretrained prednet for video prediction
model_dir = './model_data_keras2/'
prednet = load_prednet(model_dir, output_mode='prediction')

# randomly sample a given number of training videos
video_dir = '/AlgonautsVideos268_All_30fpsmax'
video_list = glob.glob(video_dir + '/*.mp4')
video_list.sort()
n_train_videos = 1000
n_predictions = 6
video_indices = np.random.choice(range(n_train_videos), n_predictions)

video_pred_dir = "/Algonauts_2021_Models/prednet/video_predictions"
if not os.path.exists(video_pred_dir):
    os.makedirs(video_pred_dir)

for i in video_indices:

    frames, num_frames = sample_video_frames(video_list[i])

    # preprocess frames
    frames = frames / 255.0
    frames = np.array([cv2.resize(frames[frame], (128, 160)) for frame in
                       range(num_frames)])
    frames = np.expand_dims(frames, axis=0)
    frames = np.transpose(frames, (0, 1, 4, 3, 2))

    # predict video frames
    batch_size = 1
    pred_frames = prednet.predict(frames, batch_size)

    # process predictions for saving
    pred_frames = np.transpose(pred_frames, axes=(0, 1, 4, 3, 2))
    pred_frames = pred_frames.squeeze(0)

    # process original frames for saving
    frames = np.transpose(frames, axes=(0, 1, 4, 3, 2))
    frames = frames.squeeze(0)

    # save paths
    actual_path = os.path.join(video_pred_dir, f"actual_video_{i}.gif")
    pred_path = os.path.join(video_pred_dir, f"predicted_video_{i}.gif")

    # make gifs
    fps = 5.33
    frame_pause = 0.75
    imageio.mimwrite(actual_path, frames, duration=frame_pause, fps)
    imageio.mimwrite(pred_path, pred_frames, duration=frame_pause, fps)

