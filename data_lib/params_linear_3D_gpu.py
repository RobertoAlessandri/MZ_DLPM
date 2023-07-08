from numba import jit, cuda
import numpy as np
import tensorflow as tf
import sfs
import matplotlib.pyplot as plt
from data_lib import soundfield_generation as sg
import os
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
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
grid2D = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=spacing)
grid3D = sfs.util.xyz_grid([-2, 2], [-2, 2], [0, 4], spacing=spacing)

array = sfs.array.linear(N_lspks, spacing, center=[-1.5, 0, 2], orientation=[1, 0, 0])

array_pos = array.x
theta_l = np.zeros(len(array_pos))
for n in range(len(array_pos)):
    _, theta_l[n] = sg.cart2pol(array_pos[n, 0], array_pos[n, 1])

N_sample = 200
x = np.linspace(-2, 2, N_sample)
grid_x, grid_y, grid_z = np.meshgrid(x, x, x)
point = np.array([grid_x.ravel(), grid_y.ravel(), 2*np.ones_like(grid_x.ravel())]).T  # np.zeros_like(grid_x.ravel())]).T
#print("point = {}, \n np.shape(point) = {}, \n np.ndim(point) = {} ".format(point, np.shape(point), np.ndim(point)))
print("np.shape(point) = {}, \n np.ndim(point) = {} ".format(np.shape(point), np.ndim(point)))

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
#print("rangeX = {}, rangeX[0] = {}, rangeX[1] = {}".format(rangeX, rangeX[0], rangeX[1]))
print("point.shape[0] = {}".format(point.shape[0]))

#@jit(target_backend='cuda')
def pointIdx(point, rangeX, rangeY_B, rangeY_D, first):
    for n_pX in range(point.shape[0]):
    #r_point, theta_point = sg.cart2pol(point[n_p, 0], point[n_p, 1])
        if (point[n_pX, 0] >= (rangeX[0])) & (point[n_pX, 0] <= (rangeX[1])):
        #print("qui entra? X")
        #for n_pY in range(point.shape[1]):
            #if true:
            if (((point[n_pX, 1] >= (rangeY_B[0])) & (point[n_pX, 1] <= (rangeY_B[1]))) | ((point[n_pX, 1] <= (rangeY_D[0])) & (point[n_pX, 1] >= (rangeY_D[1])))):
                #print("qui entra? Y")
                if first:
                    #print("if?")
                    point_lr = np.expand_dims(point[n_pX], axis=0)
                    #print("n_pX, 0 \n np.shape(point_lr) = {} ".format(np.shape(point_lr)))

                    #point_lr = np.expand_dims(point[n_pX], axis=1)
                    #print("n_px, 1 \n np.shape(point_lr) = {} ".format(np.shape(point_lr)))

                    #point_lr = np.expand_dims(point[n_pX][n_pY], axis=0)
                    #print("n_pX, n_pY, 0 \n np.shape(point_lr) = {} ".format(np.shape(point_lr)))

                    #point_lr = np.expand_dims(point[n_pX][n_pY], axis=1)
                    #print("n_pX, n_pY, 1 \n np.shape(point_lr) = {} ".format(np.shape(point_lr)))

                    idx_lr = np.expand_dims(n_pX, axis=0)

                    #idx_lr = np.expand_dims(np.asarray([n_pX, n_pY]), axis=0)
                    first = False
                else:
                    #print("or else?")

                    point_lr = np.concatenate([point_lr, np.expand_dims(point[n_pX], axis=0)])
                    idx_lr = np.concatenate([idx_lr, np.expand_dims(n_pX, axis=0)])
    return point_lr, idx_lr

point_lr, idx_lr = pointIdx(point, rangeX, rangeY_B, rangeY_D, first)

                    #point_lr = np.concatenate([point_lr, np.expand_dims(point[n_pX][n_pY], axis=0)])
                    #idx_lr = np.concatenate([idx_lr, np.expand_dims(np.asarray([n_pX, n_pY]), axis=0)])

#print("point_lr= {}, \n point_lr.shape = {} ".format(point_lr, np.shape(point_lr)))
#print("idx_lr= {}, \n idx_lr.shape = {} ".format(idx_lr, np.shape(idx_lr)))
print("\n point_lr.shape = {} ".format(np.shape(point_lr)))
print("\n idx_lr.shape = {} ".format(np.shape(idx_lr)))

# An easier way to extract Control Points
#print("point = {} , point.shape() = {}".format(point, np.shape(point)))

#point_grid_mics_X = point[np.arange(15000, 25000, 1), 0]  # was 41
#point_grid_mics_Y_B = point[np.arange(25000, 35000, 1), 1]  # was 41 #np.arange(0, 1, 5
#point_grid_mics_Y_D = point[np.arange(5000, 15000, 1), 1]  # was 41
#point_grid_mics_Z = point[np.arange(20000, 30000, 1), 2]  # was 41

#point_grid_mics_ = point[np.arange(15000, 25000, 1), :]  # was 41

#print("np.arange(-1, 1, 0.05) = {} ".format(np.arange(-1, 1, 0.05)))
#print("point_grid_mics_X = {}, elements type = {}".format(point_grid_mics_X, type(point_grid_mics_X[0])))
t_X = point[np.arange(0, 40000, 1), 0]
#print("t_X = {}, elements type = {}".format(t_X, type(t_X[0])))


#point_grid_mics_B = point[point_grid_mics_X/2, point_grid_mics_Y_B]  # , point_grid_mics_Z]
#point_grid_mics_D = point[point_grid_mics_X/2, point_grid_mics_Y_D]

#point_grids_mics = point_grid_mics_B + point_grid_mics_D
point_grids_mics = point[idx_lr]
#print("point = {}, and shape = {}".format(point, np.shape(point)))
#print("idx_CPs = {}, and shape = {}".format(idx_CPs, np.shape(idx_CPs)))
#print("point_grids_mics = {}, and shape = {}".format(point_grids_mics, np.shape(point_grids_mics)))


#plt.figure()
N_lr_pts = len(point_lr)
x = np.linspace(-2, 2, 40)  # (40, )
x, y, z = np.meshgrid(x, x, x)  # (40, 40)
x = x.ravel()  # (1600, )
y = y.ravel()
z = z.ravel()
#rho, _ = sg.cart2pol(x, y)
#rho_bool = rho < radius
#x, y = x[rho_bool], y[rho_bool]
point_cp = np.array([x, y, 2*np.ones(x.shape)]).transpose()  # (1600, 3)
print("np.shape(point_cp) = {}".format(np.shape(point_cp)))
#idx_cp = np.zeros(point_cp.shape[0], dtype=int)  # (3, )
#print("np.shape(idx_cp) = {}".format(np.shape(idx_cp)))
# ? can't understand the goal of this for cycle

@jit(target_backend='cuda')
def idxCP(point_cp, point_lr):
    idx_cp = np.zeros(point_cp.shape[0], dtype=int)  # (3, )
    for n_p in range(len(point_cp)):
        idx_cp[n_p] = np.argmin(np.linalg.norm(point_cp[n_p] - point_lr, axis=1))
    #print("n_p = {} \nidx_cp[n_p] = {} \npoint_cp[n_p] - point_lr = {} \nnp.linalg.norm(point_cp[n_p] - point_lr) = {} ".format(n_p, idx_cp[np], point_cp[n_p] - point_lr, np.linalg.norm(point_cp[n_p] - point_lr)))
    # operands could not be broadcast together with shapes (3,) (60003,)
    return idx_cp
idx_cp = idxCP(point_cp, point_lr)
print(str(len(x.ravel())), 'control points')


plot_setup_1 = False
#plot_setup = False
if plot_setup_1:
    plt.figure(figsize=(10, 10))
    plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.title("First 2D")
    plt.show()

plot_setup_1_3D = True
#plot_setup = False
if plot_setup_1_3D:
    plt.figure(figsize=(10, 10))
    plt.plot(point[:, 0], point[:, 1], point[:, 2], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], point_lr[:, 2], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], point_cp[:, 2], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], array_pos[:, 2], 'k*')
    plt.title("First 3D")
    plt.show()
#print(str(len(point_cp)) + ' control points')

################## Test sources #######################################################################################
# Generate sources
#step_radius = 0.1
#radius_sources_train = np.arange(2, 4, step_radius)
#radius_sources_test = np.arange(2 + (step_radius / 2), 4+ step_radius + (step_radius / 2), step_radius)
#n_sources_radius = 128
#src_pos_train = np.zeros((len(radius_sources_train) * n_sources_radius, 2))
#src_pos_test = np.zeros((len(radius_sources_train), n_sources_radius, 2))
start_point = -1.5
end_point = 1.5
num_sources = 16
step = (start_point - end_point) / num_sources  # 0.1875
dist_sources = np.arange(start_point, end_point, step)
sources_positions = np.zeros((len(dist_sources) * num_sources))
#line_array = np.linspace(start_point, end_point, num=num_sources, endpoint=True)

#point_grid_src_Y = point[:, np.arange(start_point, end_point, step), :]  # was 41


#point_grid_src = point[-1.5, point_grid_src_Y, 2]


#for i in range(0, len(line_array)):
  #room2D.add_source([-1.5, line_array[i]], signal=signal)
  #room3D.add_source([-1.5, line_array[i], 2], signal=signal)

#angles = np.linspace(0, 2 * np.pi, n_sources_radius)
#for n_r in range(len(radius_sources_train)):
    #for n_s in range(n_sources_radius):
        #src_pos_train[(n_r * n_sources_radius) + n_s] = sg.pol2cart(radius_sources_train[n_r], angles[n_s])
        #src_pos_test[n_r, n_s] = sg.pol2cart(radius_sources_test[n_r], angles[n_s])


plot_setup_2 = False
if plot_setup_2:
    plt.figure(figsize=(10, 10))
    #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    #plt.plot(point[idx_lr[idx_cp], 0], point[idx_lr[idx_cp], 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    #plt.plot(src_pos_train[:,0], src_pos_train[:,1],'c*')
    #plt.plot(src_pos_test[:,:,0], src_pos_test[:,:,1],'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['eval points', 'control points', 'loudspeakers'])
    plt.title("Second 2D")
    plt.show()

plot_setup_2_3D = True
#ax = plt.figure().add_subplot(projection='3d')
def plotDDD(plot_setup_2_3D, x, y, point_lr, point_cp, array_pos):
    if plot_setup_2_3D:
    #plt.figure(figsize=(10, 10))
        ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
        ax.plot(x, y, zs=0, zdir='z')
    #plt.plot(point[:, 0], point[:, 1], 'r*')
        plt.plot(point_lr[:, 0], point_lr[:, 1], point_lr[:, 2], 'g*')
        plt.plot(point_cp[:, 0], point_cp[:, 1], point_cp[:, 2], 'b*')
    #plt.plot(point[idx_lr[idx_cp], 0], point[idx_lr[idx_cp], 1], 'b*')
        plt.plot(array_pos[:, 0], array_pos[:, 1], array_pos[:, 2], 'k*')
    #plt.plot(src_pos_train[:,0], src_pos_train[:,1],'c*')
    #plt.plot(src_pos_test[:,:,0], src_pos_test[:,:,1],'r*')
        plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
        plt.legend(['eval points', 'control points', 'loudspeakers'])
        ax.set_zlim(0, 4)
    #ax.view_init(elev=2., azim=0, roll=0)
        plt.title("Second 3D")
        plt.show()
plotDDD(plot_setup_2_3D, x, y, point_lr, point_cp, array_pos)

print("params_linear_3D Ended")