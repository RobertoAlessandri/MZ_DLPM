import numpy as np

def normalize(x):
    x_mean = np.mean(x)
    x_std_dev = np.sqrt(np.var(x))
    return (x - x_mean) / x_std_dev, x_mean, x_std_dev


def denormalize(x_norm, x_mean, x_std_dev):
    x_denorm = (x_norm * x_std_dev) + x_mean
    return x_denorm
