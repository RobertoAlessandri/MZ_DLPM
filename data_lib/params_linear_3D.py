#from numba import jit, cuda
import numpy as np
import tensorflow as tf
import sfs
import matplotlib.pyplot as plt
from data_lib import soundfield_generation as sg
import os
#os.environ['CUDA_VISIBLE_DEVICES']=''
c_complex = 343
pi_complex = np.pi

# Soundfield params (this is not right place)
nfft = 128  # Number of fft points
d = 0.02  # Spacing between sensors [m]
c = 343  # sound speed at 20 degrees [m/s]
f_s = 3200  # Maximum frequency to be considered in Hz
s_r = 2 * f_s  # Sampling rate
# Frequency axis
f_axis = np.fft.rfftfreq(nfft, 1/s_r)
f_axis = f_axis[1:]
N_freqs = len(f_axis)
wc = 2 * np.pi * f_axis
N = 41  # ?

#plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"],
    #'font.size': 20})

# Linear array parameters ###########################################################################################
N_lspks = 16  # Numbgrider of loudspeakers
spacing = 0.2
grid3D = sfs.util.xyz_grid([-2, 2], [-2, 2], [0, 4], spacing=spacing)

array = sfs.array.linear(N_lspks, spacing, center=[-1.5, 0, 2], orientation=[1, 0, 0])

array_pos = array.x
theta_l = np.zeros(len(array_pos))
for n in range(len(array_pos)):
    _, theta_l[n] = sg.cart2pol(array_pos[n, 0], array_pos[n, 1])

N_sample = 128
x = np.linspace(-2, 2, N_sample)
z = np.linspace(0, 4, N_sample)
grid_x, grid_y, grid_z = np.meshgrid(x, x, z)
point = np.array([grid_x.ravel(), grid_y.ravel(), 2*np.ones_like(grid_z.ravel())]).T  # np.zeros_like(grid_x.ravel())]).T
print("np.shape(point) = {}, \nnp.ndim(point) = {} ".format(np.shape(point), np.ndim(point)))

N_pts = len(grid_x.ravel())

# Extract points corresponding to interior field w.r.t. the array
first = True
# Delimiting corners of Control Points  #
#cornersB = np.asarray([[-0.25, 0.75], [0.25, 0.75], [0.25, 0.25], [-0.25, 0.25]])
#cornersD = np.asarray([[-0.25, -0.25], [0.25, -0.25], [0.25, -0.75], [-0.25, -0.75]])
rangeX = np.asarray([-0.25, 0.25])
rangeY_B = np.asarray([0.25, 0.75])
rangeY_D = np.asarray([-0.25, -0.75])
rangeZ = np.asarray([1.75, 2.25])
true = True;
#@jit(target_backend='cuda')
#def pointIdx(point, rangeX, rangeY_B, rangeY_D, first):
for n_pX in range(point.shape[0]):
    #r_point, theta_point = sg.cart2pol(point[n_p, 0], point[n_p, 1])
    if (point[n_pX, 0] >= (rangeX[0])) & (point[n_pX, 0] <= (rangeX[1])):
        if (((point[n_pX, 1] >= (rangeY_B[0])) & (point[n_pX, 1] <= (rangeY_B[1]))) | ((point[n_pX, 1] <= (rangeY_D[0])) & (point[n_pX, 1] >= (rangeY_D[1])))):
            #if((point[n_pX, 2] >= rangeZ[0]) & (point[n_pX, 2] <= rangeZ[1])):
            if(2 > 1):   #print("qui entra? Y")
                if first:
                    point_lr = np.expand_dims(point[n_pX], axis=0)
                    idx_lr = np.expand_dims(n_pX, axis=0)
                    first = False
                else:
                    point_lr = np.concatenate([point_lr, np.expand_dims(point[n_pX], axis=0)])
                    idx_lr = np.concatenate([idx_lr, np.expand_dims(n_pX, axis=0)])
    #return point_lr, idx_lr
print("point_lr.shape = {} ".format(np.shape(point_lr)))
print("idx_lr.shape = {} ".format(np.shape(idx_lr)))

# An easier way to extract Control Points
point_grids_mics = point[idx_lr]
N_lr_pts = len(point_lr)
sample_cp = 128
x = np.linspace(-0.25, 0.25, 8)  # (40, )
yb = np.linspace(0.25, 0.75, 8)  # (40, )
yd = np.linspace(-0.75, -0.25, 8)  # (40, )
y = np.concatenate((yd, yb))
z = np.linspace(1.75, 2.25, 8)
x, y, z = np.meshgrid(x, y, z)  # (40, 40)
x = x.ravel()  # (1600, )
y = y.ravel()
z = z.ravel()
idx_cp = np.arange(0, len(point_lr), sample_cp)  # relative to listening area
point_cp = point_lr[idx_cp]
print("len(idx_cp) = ", len(idx_cp))
point_cp_temp = np.array([x, y, 2*np.ones(x.shape)]).transpose()  # (1600, 3) RIPRISTINA !!!
print("np.shape(point_lr) = {}\nnp.shape(point_cp) = {}".format(np.shape(point_lr), np.shape(point_cp)))
for n_pX in range(point_cp_temp.shape[0]):
    if (point_cp_temp[n_pX, 0] >= (rangeX[0])) & (point_cp_temp[n_pX, 0] <= (rangeX[1])):
        if (((point_cp_temp[n_pX, 1] >= (rangeY_B[0])) & (point_cp_temp[n_pX, 1] <= (rangeY_B[1]))) | ((point_cp_temp[n_pX, 1] <= (rangeY_D[0])) & (point_cp_temp[n_pX, 1] >= (rangeY_D[1])))):
            if first:
                point_cp = np.expand_dims(point_cp_temp[n_pX], axis=0)
                idx_cp = np.expand_dims(n_pX, axis=0)
                first = False
            else:
                point_cp = np.concatenate([point_cp, np.expand_dims(point_cp_temp[n_pX], axis=0)])
                idx_cp = np.concatenate([idx_cp, np.expand_dims(n_pX, axis=0)])
for n_p in range(len(point_cp)):
    idx_cp[n_p] = np.argmin(np.linalg.norm(point_cp[n_p] - point_lr, axis=1))
    #print("n_p = {} \nidx_cp[n_p] = {} \npoint_cp[n_p] - point_lr = {} \nnp.linalg.norm(point_cp[n_p] - point_lr) = {} ".format(n_p, idx_cp[np], point_cp[n_p] - point_lr, np.linalg.norm(point_cp[n_p] - point_lr)))
print(str(len(x.ravel())), 'control points')

step_radius = 0.2
radius_sources_train = np.arange(1, 3, step_radius)
radius_sources_test = np.arange(1 + (step_radius / 2), 3 + step_radius + (step_radius / 2), step_radius)
n_sources_radius = 128
src_pos_train = np.zeros((len(radius_sources_train) * n_sources_radius, 2))  # should be 3
src_pos_test = np.zeros((len(radius_sources_train), n_sources_radius, 2))  # should be 3

angles = np.linspace(3 * np.pi / 4, 5 * np.pi / 4, n_sources_radius)
#for n_r in range(len(radius_sources_train)):
    #for n_s in range(n_sources_radius):
        #for n_ss in range(n_sources_radius):
            #src_pos_train[(n_r * n_sources_radius) + n_s + n_ss] = sg.pol2cartDDD(radius_sources_train[n_r], angles[n_s], angles[n_ss])
            #src_pos_test[n_r, n_s, n_ss] = sg.pol2cartDDD(radius_sources_test[n_r], angles[n_s], angles[n_ss])

for n_r in range(len(radius_sources_train)):
    for n_s in range(n_sources_radius):
        src_pos_train[(n_r * n_sources_radius) + n_s] = sg.pol2cart(radius_sources_train[n_r], angles[n_s])
        src_pos_test[n_r, n_s] = sg.pol2cart(radius_sources_test[n_r], angles[n_s])


plot_setup_1 = False
#plot_setup = False
if plot_setup_1:
    plt.figure(figsize=(10, 10))
    plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title("First 2D")
    plt.show()

plot_setup_1_3D = False
#plot_setup = False
if plot_setup_1_3D:
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    plt.plot(point[:, 0], point[:, 1], point[:, 2], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], point_lr[:, 2], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], point_cp[:, 2], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], array_pos[:, 2], 'k*')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 4)
    plt.title("First 3D")
    plt.show()
#print(str(len(point_cp)) + ' control points')

################## Test sources #######################################################################################
# Generate sources
start_point = -1.5
end_point = 1.5
num_sources = 16
step = (start_point - end_point) / num_sources  # 0.1875
dist_sources = np.arange(start_point, end_point, step)
sources_positions = np.zeros((len(dist_sources) * num_sources))
#for i in range(0, len(line_array)):
  #room2D.add_source([-1.5, line_array[i]], signal=signal)
  #room3D.add_source([-1.5, line_array[i], 2], signal=signal)

#angles = np.linspace(0, 2 * np.pi, n_sources_radius)
#for n_r in range(len(radius_sources_train)):
    #for n_s in range(n_sources_radius):
        #src_pos_train[(n_r * n_sources_radius) + n_s] = sg.pol2cart(radius_sources_train[n_r], angles[n_s])
        #src_pos_test[n_r, n_s] = sg.pol2cart(radius_sources_test[n_r], angles[n_s])

plot_setup_2 = True
if plot_setup_2:
    plt.figure(figsize=(10, 10))
    #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    #plt.plot(point[idx_lr[idx_cp], 0], point[idx_lr[idx_cp], 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.plot(src_pos_train[:,0], src_pos_train[:,1],'c*')
    plt.plot(src_pos_test[:,:,0], src_pos_test[:,:,1],'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend(['eval points', 'control points', 'loudspeakers'])
    plt.title("Second 2D")
    plt.show()

plot_setup_2_3D = True
#ax = plt.figure().add_subplot(projection='3d')

if plot_setup_2_3D:
    #plt.figure(figsize=(10, 10))
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    ax.plot(x, y, zs=0, zdir='z')
    #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], point_lr[:, 2], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], point_cp[:, 2], 'b*')
    #plt.plot(point[idx_lr[idx_cp], 0], point[idx_lr[idx_cp], 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], array_pos[:, 2], 'k*')
    #plt.plot(src_pos_train[:, 0], src_pos_train[:, 1], src_pos_train[:, 2], 'c*')
    plt.plot(src_pos_train[:, 0], src_pos_train[:, 1], 2, 'c*')
    #plt.plot(src_pos_test[:, :, 0], src_pos_test[:, :, 1], src_pos_test[:, :, 2], 'r*')
    #plt.plot(src_pos_test[:, :, 0], src_pos_test[:, :, 1], 2, 'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['eval points', 'control points', 'loudspeakers', 'train sources', 'test sources'])
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 4)
    #ax.view_init(elev=2., azim=0, roll=0)
    plt.title("Second 3D")
    plt.show()

print("##   params_linear_3D Ended  ##")