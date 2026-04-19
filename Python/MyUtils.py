import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def to_tuple(x):
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
        return tuple(arr.ravel())  # Ensure 1D flattening
    if isinstance(x, np.ndarray):
        return tuple(x.ravel())    # Flatten similarly
    if isinstance(x, (int, float)):
        return (x,)                # Wrap scalar
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x,)


def random_local_rotation(device, dtype):
    """
    Generate a random SO(3) rotation matrix:
    - yaw   (z-axis): [0, 2π)
    - pitch (y-axis): [-15°, +15°]
    - roll  (x-axis): [-15°, +15°]
    """
    # Angle ranges
    yaw   = random.uniform(0.0, 2.0 * math.pi)
    pitch = random.uniform(-15.0, 15.0) * math.pi / 180.0
    roll  = random.uniform(-15.0, 15.0) * math.pi / 180.0

    cy, sy = math.cos(yaw),   math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll),  math.sin(roll)

    # Rz(yaw)
    Rz = torch.tensor([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ], device=device, dtype=dtype)

    # Ry(pitch)
    Ry = torch.tensor([
        [ cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp]
    ], device=device, dtype=dtype)

    # Rx(roll)
    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0,  cr, -sr],
        [0.0,  sr,  cr]
    ], device=device, dtype=dtype)

    # Rotation order: roll → pitch → yaw
    R = Rz @ Ry @ Rx
    return R