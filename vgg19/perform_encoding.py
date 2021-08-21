import numpy as np
from utils import load_fmri, load_activations, vectorized_correlation
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


def perform_encoding(pca_dir, fmri_dir, layer, sub, ROI):
    """This function linearly regresses pca-reduced features/activations of a
    given layer of a neural network to the fMRI activity of a given brain region
    (ROI) in a given subject. The fitted model is used to predict part of the
    training set (validation mode). The model predictions are saved in a specified
    directory.

    # Arguments
    -------------
    pca_dir : str
        path to PCA features.
    fmri_dir : str
        path to fMRI data.
    results_dir : str
        saving directory for results.
    layer : str
        layer name from which to extract activations.
    sub : str
        participant number path.
    ROI : str
        region of interest (brain region) from which to extract fMRI data.
    batch_size : int
        Number of voxels processed when fitting the linear regressor. 1000 by
        default.

    """

    # load train activations
    train_activations = load_activations(pca_dir, layer)

    # load fMRI data for a given subject and ROI
    fmri_train_all = load_fmri(fmri_dir, sub, ROI)
    # get number of voxels for given ROI
    num_voxels = fmri_train_all.shape[1]

    # create training and validation sets
    val_activations = train_activations[900:,:]
    train_activations = train_activations[:900,:]
    fmri_train = fmri_train_all[:900,:]
    fmri_val = fmri_train_all[900:,:]

    # initialise results - #validation videos x #voxels
    pred_fmri = np.zeros_like(fmri_val)

    # perform multiple multivariate regression
    reg = MultiOutputRegressor(LinearRegression())
    reg.fit(train_activations, fmri_train)
    pred_fmri = reg.predict(val_activations)

    # correlation between validation predictions and ground truth
    corr = vectorized_correlation(fmri_val, pred_fmri)
    print("----------------------------------------------------------------------------")
    # get mean correlation
    voxelwise_corr = round(corr.mean(), 6)
    print("Mean correlation for ROI : " ,ROI ," in " ,sub ," using " ,layer
          ," is :", voxelwise_corr)

    return voxelwise_corr
