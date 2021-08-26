from load_fmri import load_fmri
from vectorised_correlation import vectorised_correlation
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


def perform_encoding(train_features, fmri_dir, sub, ROI):
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
