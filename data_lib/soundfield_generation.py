import numpy as np
import tqdm
#from data_lib import params_linear_2D
#from data_lib import params_circular
import scipy

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def cart2polDDD(x, y, z):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, y)  #
    return rho, phi, theta

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def pol2cartDDD(rho, phi, theta):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = rho * np.sin(theta)  #
    return x, y, z

print("Sound_Field_Generation Ended")
