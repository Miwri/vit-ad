"""A module which provides diverse helper functions"""

from __future__ import annotations

from torch import cuda, nn

BIAS_FILL = 0.001


# TODO this is not working as it deletes only the object in the method
def clear_gpu(cuda_obj: any, device: str):
    """clear gpu of unused data to avoid memory overflow"""
    if device != "cpu":
        cuda_obj.cpu()
        del cuda_obj
        cuda.empty_cache()


def init_weights(m):
    """Function to init weights with xavier normal and add bias to a cnn or linear parameter of nn.Module"""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(BIAS_FILL)
