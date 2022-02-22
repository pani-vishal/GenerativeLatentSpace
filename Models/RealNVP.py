# ------------------------------------------------------------------------------
# PyTorch implementation of a convolutional Real NVP (2017 L. Dinh
# "Density estimation using Real NVP" in https://arxiv.org/abs/1605.08803)
# ------------------------------------------------------------------------------

import os

import matplotlib.pyplot as plt
import numpy as np
import torch as pt


import sys
sys.path.append('./')
from Models.base import LinBlock


