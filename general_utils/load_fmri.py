import os
import pickle
import numpy as np


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
