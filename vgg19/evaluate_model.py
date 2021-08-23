import glob
import numpy as np
from utils import load_weights
from extract_features import extract_activations, apply_pca
from perform_encoding import perform_encoding


# subjects, regions of interest, model and layer
subs = ["sub01", "sub02", "sub03", "sub04", "sub05", "sub06", "sub07", "sub08", "sub09", "sub10"]
ROIs = ["V1", "V2", "V3", "V4", "LOC", "EBA", "FFA", "STS", "PPA"]
vgg19_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
layer = 'layer_19'

# video paths
video_dir = '/AlgonautsVideos268_All_30fpsmax'
video_list = glob.glob(video_dir + '/*.mp4')
video_list.sort()
train_videos = 1000

# load pretrained model
vgg19 = load_weights(vgg19_url)

# extract activations
activations_dir = "/activations_vgg19"
extract_activations(vgg19, video_list[:train_videos], activations_dir, layer)

# perform pca on activations
pca_dir = '/pca_activations'
apply_pca(activations_dir, pca_dir, layer)

fmri_dir = '/participants_data_v2021/mini_track'
results_dir = '/predictions_vgg19'

# compute voxelwise correlations for all subjects and ROIs
voxelwise_corrs = np.zeros((len(subs), len(ROIs)))
for i, sub in enumerate(subs):
    for j, ROI in enumerate(ROIs):
        voxelwise_corrs[i, j] = perform_encoding(pca_dir,
                                                 fmri_dir,
                                                 layer=layer,
                                                 sub=sub,
                                                 ROI=ROI)

np.save(results_dir, voxelwise_corrs)

