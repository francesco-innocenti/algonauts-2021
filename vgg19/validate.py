import glob
import numpy as np
from utils import load_weights, extract_activations, apply_PCA, perform_encoding

# subjects, regions of interest, model weights and layer
subs = ["sub01", "sub02", "sub03", "sub04", "sub05", "sub06", "sub07", "sub08", "sub09", "sub10"]
ROIs = ["V1", "V2", "V3", "V4", "LOC", "EBA", "FFA", "STS", "PPA"]
model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
layer = 'layer_19'

video_dir = '/AlgonautsVideos268_All_30fpsmax'
activations_dir = "/activations_vgg19"
pca_dir = '/pca_activations'
fmri_dir = '/participants_data_v2021/mini_track'
predictions_dir = '/predictions_vgg19'

# video paths
video_list = glob.glob(video_dir + '/*.mp4')
video_list.sort()

# load pretrained model
vgg19 = load_weights(model_url)

# extract features and apply pca
extract_activations(vgg19, video_list, activations_dir, layer=19)  # specify layer
apply_PCA(activations_dir, pca_dir)

# compute voxelwise correlations for all subjects and ROIs
voxelwise_corrs = np.zeros((len(subs), len(ROIs)))
for i, sub in enumerate(subs):
    for j, ROI in enumerate(ROIs):
        voxelwise_corrs[i, j] = perform_encoding(pca_dir,
                                                 fmri_dir,
                                                 predictions_dir,
                                                 layer=layer,
                                                 sub=sub,
                                                 ROI=ROI)

np.save(predictions_dir, voxelwise_corrs)

