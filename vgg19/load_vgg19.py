import torch
from torch import hub
from vgg19 import VGG19


def load_vgg19(model_url):
    """This function loads pretrained weights onto a neural network (VGG19).

    Args:
        model_url: str
            URL of pretrained pytorch model.

    Returns:
        model: class
            pytorch model ready for inference.
    """

    model = VGG19()
    param_names = list(model.state_dict())
    model_dict = {k: None for k in param_names}
    state_dict = hub.load_state_dict_from_url(model_url)

    i = 0
    for v in state_dict.values():
        model_dict[param_names[i]] = v
        i += 1

    model.load_state_dict(model_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    return model
