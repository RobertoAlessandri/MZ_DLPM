import numpy as np
import matplotlib.pyplot as plt

gt = np.load('/pressure_matching_deep_learning/dataset/linear_array/gt_soundfield_train.npy"')

#gt = np.load('C:\Users\rales\OneDrive\Desktop\POLIMI\TESI\pressure_matching_deep_learning\dataset\linear_array\gt_soundfield_train.npy"')
gf = np.load('C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/dataset/train/linear_array/green_function_sec_sources_nl_16_r_-0.25.npy')
print("shape gt = {}\nshape gf = {}".format(np.shape(gt), np.shape(gf)))

#plt.figure(figsize=(10, 10))
#plt.plot(point[:, 0], point[:, 1], 'r*')
#plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
#plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
#plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
#plt.title("First 2D")
#plt.show()