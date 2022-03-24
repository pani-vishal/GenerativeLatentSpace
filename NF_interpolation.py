# ------------------------------------------------------------------------------
# Interpolation in latent space, by default for MNIST digits, but can be changed
# easily.
# ------------------------------------------------------------------------------

import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import os
import gc
import time

from argparse import ArgumentParser

import sys
sys.path.append('./')
from Models.RealNVP import *
from Geometry.geodesic import trainGeodesic
from Geometry.curves import BezierCurve


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('gen', help="Name of generator architecture in 'Models/' directory.", choices=['RealNVP'])
    parser.add_argument('--trained_gen', help="Name of trained generator in 'TrainedModels/' directory.", type=str, default=None)
    parser.add_argument('--latent_dim', help="Dimension of latent space.", type=int, default=2)
    parser.add_argument('--M_batch_size', help="Batchsize for computation of metric.", type=int, default=1)
    parser.add_argument('--epochs', help="Number of epochs to train the shorter curve.", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists("Outputs"):
        os.makedirs("Outputs")

    device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')
    
    # Discretization of geodesic curve
    N_t = 20
    bc0 = -pt.ones(args.latent_dim)
    bc1 = pt.ones(args.latent_dim)

    if args.gen == "RealNVP":
        modelG = RealNVP(latent_dim=args.latent_dim)
        args.trained_gen = args.trained_gen if args.trained_gen is not None else "flow.pth"

    modelG.load_state_dict(pt.load(os.path.join("TrainedModels", args.trained_gen)))
    modelG.to(device)
    modelG.eval()
    print("Generator loaded!")

    ### Find shorter path than straight line
    print("Optimizing for shorter path...")
    best_gamma, length_history = trainGeodesicNF(bc0, bc1, N_t, )

