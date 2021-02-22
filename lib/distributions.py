import torch
import numpy as np

Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def log_standard_normal(x):
    z = - 0.5 * Log2PI
    return - x ** 2 / 2 + z
