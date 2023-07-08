import numpy as np
import sfs
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

import matplotlib.pyplot as plt
from matplotlib import cm, colors, patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm


def ReLU(size, a, b):
  x = np.random.rand(size,1)-0.5
  y = np.zeros(np.shape(x))
  for i in range(len(x)):
    if x[i]>0:
      y[i] = x[i]
    else:
      y[i]=0
  #y = y + 0.2*(np.random.rand(*x.shape)-0.5)
  y = y #+ 0.05*(np.random.randn(*x.shape))
  return x, y

def PReLU(size, alpha):
  x = np.random.rand(size,1)-0.5
  y = np.zeros(np.shape(x))
  for i in range(len(x)):
    if x[i]>0:
      y[i] = x[i]
    else:
      y[i] = x[i] * alpha
  #y = y + 0.2*(np.random.rand(*x.shape)-0.5)
  y = y #+ 0.05*(np.random.randn(*x.shape))
  return x, y

pReLU = False
if pReLU:
    alpha = []
    #plt.figure()
    for a in range(1, 10):
        alpha.append(a/10)
        Xdata, Ydata = PReLU(1000, alpha[a-1])
        plt.scatter(Xdata,Ydata);
    plt.legend(['a={}'.format(alpha[0]), 'a={}'.format(alpha[1]), 'a={}'.format(alpha[2]), 'a={}'.format(alpha[3]), 'a={}'.format(alpha[4]), 'a={}'.format(alpha[5]), 'a={}'.format(alpha[6]), 'a={}'.format(alpha[7]), 'a={}'.format(alpha[8])])
    plt.xlabel('x')
    plt.ylabel('max(0,x)+a*min(0,x)')
    plt.title('PReLU')
    plt.show()

reLU = False
if reLU:
    Xdata, Ydata = ReLU(1000, 0.8, 1.3)
    plt.scatter(Xdata,Ydata);
    plt.xlabel('x')
    plt.ylabel('max(0,x)')
    plt.title('ReLU')


sphHarm = False
if sphHarm:
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    m_max, n_max = 2, 3
    for n in range(n_max+1):
        for m in range(n+1):

            # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
            fcolors = sph_harm(m, n, theta, phi).imag
            fmax, fmin = 0.341, 0.34#fcolors.max(), fcolors.min()
            fcolors = (fcolors - fmin)/(fmax - fmin)
#0.34
            # Set the aspect ratio to 1 so our sphere looks spherical
            fig = plt.figure(figsize=plt.figaspect(1.))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.seismic(fcolors))
            # Turn off the axis planes
            ax.set_title('Spherical Harmonic of order {} and degree {}'.format(m, n))
            ax.set_axis_off()
            plt.show()

Bessel = False
if Bessel:
    x = np.arange(0.0, 10.0, 0.01)
    fig, ax = plt.subplots()
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(r'Spherical Bessel functions $j_n$')
    for n in np.arange(0, 4):
        ax.plot(x, spherical_jn(n, x), label=rf'$j_{n}$')
    plt.legend(loc='best')
    plt.show()

    x = np.arange(0.0, 10.0, 0.01)
    fig, ax = plt.subplots()
    ax.set_ylim(-2.0, 1.0)
    ax.set_title(r'Spherical Bessel functions $y_n$')
    for n in np.arange(0, 4):
        ax.plot(x, spherical_yn(n, x), label=rf'$y_{n}$')
    plt.legend(loc='best')
    plt.show()

print("Beginning params_linear_2D")

# Method to plot soundfield
def sound_field(d, selection, secondary_source, array, grid, xs, idx_x, idx_y, tapering=True, DDD=False, norm=True, db=False, dark=False):
    if tapering:
        # selection = a boolean array containing True for active loudspeakers
        tapering_window = sfs.tapering.tukey(selection, alpha=0.3)  # weights for the driving function
    else:
        tapering_window = sfs.tapering.none(selection)
    # Compute sound field for a generic driving functionsS
    # INPUT = driving function, weights, secondary source distribution, secondary source function
    # OUTPUT = Sound Field for a generic d (sound pressure values, ignores imaginary part)
    p = sfs.fd.synthesize(d, tapering_window, array, secondary_source, grid=grid)
    p_idx = sfs.fd.synthesize(d, tapering_window, array, secondary_source, grid=grid)  # contains a representation of

    if dark:# ideal soundfield with dark zone
        for i in range(len(p[0])):
            if i in idx_x:
                for j in range(len(p[1])):
                    if j in idx_y[:int(len(idx_y)/2)]:
                        #print("i, j = {}, {}".format(i, j))
                        p_idx[-j][i] = 0 + np.finfo(dtype=np.float16).eps

    #rect1 = patches.Rectangle((-0.25, 0.25), 0.5, 0.5, color='black', fc='none', lw=1)
    #rect2 = patches.Rectangle((-0.25, -0.75), 0.5, 0.5, color='black', fc='none', lw=1)

    # if 3D case
    if (DDD):
        sfs.plot3d.secondary_sources(p_idx, grid[0], [0, 0, 0])
        sfs.plot3d.secondary_sources(array.x, array.n, tapering_window)
    else:
        if norm:
            if db:
                im_db = sfs.plot2d.level(p_idx, grid, xnorm=[0, 0, 0], cmap='coolwarm')  # the normalization is respect to the point
                ##im_db.add_patch(rect1)
                ##im_db.add_patch(rect2)
                sfs.plot2d.add_colorbar(im_db)
            else:
                im = sfs.plot2d.amplitude(p_idx, grid, xnorm=[0, 0, 0])  # the normalization is respect to the point
                ##im.add_patch(rect1)
                ##im.add_patch(rect2)
                sfs.plot2d.add_colorbar(im)
        else:
            if db:
                im_db = sfs.plot2d.level(p_idx, grid, cmap='coolwarm')
                #im_db.add_patch(rect1)
                #im_db.add_patch(rect2)
                sfs.plot2d.add_colorbar(im_db)
            else:
                im = sfs.plot2d.amplitude(p_idx, grid)
                #im.add_patch(rect1)
                #im.add_patch(rect2)
                sfs.plot2d.add_colorbar(im)

        sfs.plot2d.loudspeakers(array.x, array.n, tapering_window)
        sfs.plot2d.virtualsource(xs)

    return p, p_idx

# Soundfield params (this is not right place)
nfft = 128  # Number of fft points
c = 343  # sound speed at 20 degrees [m/s]

#plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"],
    #'font.size': 20})

# Linear array parameters ###########################################################################################
N_lspks = 64  # Number of loudspeakers
array_length = 3  # [m]
spacing = array_length / N_lspks
alias_breaker = 1  # parameter to decide the percentage at which we try to surpass the aliasing restriction
f_s = (c / (2 * spacing)) * alias_breaker  # Maximum frequency to be considered in Hz to avoid spatial aliasing in
                                           # reproduction
f_s = 1500
s_r = 2 * f_s  # Sampling rate
# Frequency axis
f_axis = np.fft.rfftfreq(nfft, 1/s_r)
f_axis = f_axis[1:]
N_freqs = len(f_axis)
wc = 2 * np.pi * f_axis  # angular frequencies  # NEEDED?
wavelength = c / f_s  # minimum wavelength to be considered in [m] to avoid spatial aliasing in reproduction and
                      # acquisition
d = wavelength / 2  #  minimum distance between loudspeakers or microphones to avoid spatial aliasing
d=0.063

# Defining the Grid
grid2D = sfs.util.xyz_grid([-2, 2], [-2, 2], 2, spacing=0.02)

mingrid = sfs.util.xyz_grid([-0.25, 0.25], [-0.5, 0.5], 2, spacing=0.02)  # new grid containing only the points that
                                                                        # represent the evaluation zone
mingrid_B = sfs.util.xyz_grid([-0.25, 0.25], [0.25, 0.75], 2, spacing=0.02, endpoint=True)  # grid for bright zone
mingrid_D = sfs.util.xyz_grid([-0.25, 0.25], [-0.75, -0.25], 2, spacing=0.02, endpoint=True)  # grid for dark zone

mingrid_B_expanded = sfs.util.xyz_grid([-0.3, 0.3], [0.2, 0.8], 2, spacing=0.02, endpoint=True)  # grid for bright zone
mingrid_D_expanded = sfs.util.xyz_grid([-0.3, 0.3], [-0.8, -0.2], 2, spacing=0.02, endpoint=True)  # grid for dark zone
mingrid_expanded = sfs.util.xyz_grid([-0.3, 0.3], [-0.55, 0.55], 2, spacing=0.02)  # new grid containing only the points that


mingrid_D_switched = sfs.util.xyz_grid([-0.25, 0.25], [0.25, 0.75], 2, spacing=0.02, endpoint=True)  # grid for bright zone
mingrid_B_switched = sfs.util.xyz_grid([-0.25, 0.25], [-0.75, -0.25], 2, spacing=0.02, endpoint=True)  # grid for dark zone

# Return linear, equidistantly sampled secondary source distribution.
# Positions, orientations and weights of secondary sources.
array = sfs.array.linear(N_lspks, spacing, center=[-1.5, 0, 2], orientation=[1, 0, 0])
array_pos = array[0]
array_x = array_pos[:, 0][0]  # fixed y cooridnate
array_ori = array[1]  # orientations
array_wei = array[2]  # weights

N_sample = 128
x = np.linspace(-2, 2, N_sample)
grid_x, grid_y = np.meshgrid(x, x)
point = np.array([grid_x.ravel(), grid_y.ravel(), 2*np.ones_like(grid_x.ravel())]).T  # np.zeros_like(grid_x.ravel())]).T
N_pts = len(grid_x.ravel())

# Extract points corresponding to interior field w.r.t. the array
first = True
# Delimiting borders of Control Points  #
rangeX = np.asarray([-0.25, 0.25])
rangeY_B = np.asarray([0.25, 0.75])
rangeY_D = np.asarray([-0.25, -0.75])
rangeZ = np.asarray([1.75, 2.25])

rangeX_expanded = np.asarray([-0.3, 0.3])
rangeY_B_expanded = np.asarray([0.2, 0.8])
rangeY_D_expanded = np.asarray([-0.2, -0.8])
rangeZ_expanded = np.asarray([1.7, 2.3])

# This for cycle is not used in calculations, but is needed to plot the evaluation points
for n_pX in range(point.shape[0]):
    if (point[n_pX, 0] >= (rangeX[0])) & (point[n_pX, 0] <= (rangeX[1])):
        if (((point[n_pX, 1] >= (rangeY_B[0])) & (point[n_pX, 1] <= (rangeY_B[1]))) | ((point[n_pX, 1] <= (rangeY_D[0])) & (point[n_pX, 1] >= (rangeY_D[1])))):
            if first:
                point_lr = np.expand_dims(point[n_pX], axis=0)
                idx_lr = np.expand_dims(n_pX, axis=0)
                first = False
            else:
                point_lr = np.concatenate([point_lr, np.expand_dims(point[n_pX], axis=0)])
                idx_lr = np.concatenate([idx_lr, np.expand_dims(n_pX, axis=0)])

first = True
idx_lr_gd_x = []
idx_lr_gd_y = []

idx_lr_gd_x_expanded = []
idx_lr_gd_y_expanded = []

# Generating array of indexes that localize the x and y axes coordinates of evaluation points
for n_pX in range(grid2D[0].shape[1]):
    if (grid2D[0][0][n_pX] >= (rangeX[0])) & (grid2D[0][0][n_pX] <= (rangeX[1])):
        idx_lr_gd_x.append(n_pX)
        if (((grid2D[1].T[0][n_pX] >= (rangeY_B[0])) & (grid2D[1].T[0][n_pX] <= (rangeY_B[1]))) | ((grid2D[1].T[0][n_pX] <= (rangeY_D[0])) & (grid2D[1].T[0][n_pX] >= (rangeY_D[1])))):
            if first:
                point_lr_gd = np.expand_dims(grid2D[n_pX], axis=0)
                #print("idx_lr_gd = np.expand_dims(n_pX, axis=0) = {}".format(idx_lr_gd = np.expand_dims(n_pX, axis=0)))
                idx_lr_gd = np.expand_dims(n_pX, axis=0)
                first = False
            else:
                point_lr_gd = np.concatenate([point_lr_gd, np.expand_dims(grid2D[n_pX], axis=0)])
                idx_lr_gd = np.concatenate([idx_lr_gd, np.expand_dims(n_pX, axis=0)])

for n_pY in range(grid2D[1].shape[0]):
    if (((grid2D[1].T[0][n_pY] >= (rangeY_B[0])) & (grid2D[1].T[0][n_pY] <= (rangeY_B[1]))) | ((grid2D[1].T[0][n_pY] <= (rangeY_D[0])) & (grid2D[1].T[0][n_pY] >= (rangeY_D[1])))):
        idx_lr_gd_y.append(n_pY)  # "-1" serve per evitare che ci sia una traslazione verticale di 1 cp. Senza

# Again for ther expanded region
for n_pX in range(grid2D[0].shape[1]):
    if (grid2D[0][0][n_pX] >= (rangeX_expanded[0])) & (grid2D[0][0][n_pX] <= (rangeX_expanded[1])):
        idx_lr_gd_x_expanded.append(n_pX)
        #if (((grid2D[1].T[0][n_pX] >= (rangeY_B_expanded[0])) & (grid2D[1].T[0][n_pX] <= (rangeY_B_expanded[1]))) | ((grid2D[1].T[0][n_pX] <= (rangeY_D_expanded[0])) & (grid2D[1].T[0][n_pX] >= (rangeY_D_expanded[1])))):
            #if first:
                #point_lr_gd = np.expand_dims(grid2D[n_pX], axis=0)
                ##print("idx_lr_gd = np.expand_dims(n_pX, axis=0) = {}".format(idx_lr_gd = np.expand_dims(n_pX, axis=0)))
                #idx_lr_gd_expanded = np.expand_dims(n_pX, axis=0)
                #first = False
            #else:
                #point_lr_gd = np.concatenate([point_lr_gd, np.expand_dims(grid2D[n_pX], axis=0)])
                #idx_lr_gd_expanded = np.concatenate([idx_lr_gd_expanded, np.expand_dims(n_pX, axis=0)])

for n_pY in range(grid2D[1].shape[0]):
    if (((grid2D[1].T[0][n_pY] >= (rangeY_B_expanded[0])) & (grid2D[1].T[0][n_pY] <= (rangeY_B_expanded[1]))) | ((grid2D[1].T[0][n_pY] <= (rangeY_D_expanded[0])) & (grid2D[1].T[0][n_pY] >= (rangeY_D_expanded[1])))):
        idx_lr_gd_y_expanded.append(n_pY)  # "-1" serve per evitare che ci sia una traslazione verticale di 1 cp. Senza

# From here to line 171 is not used in calculations, and is just used to plot control points
sample_cp = 162
cps = 8
cps_min = int(abs(rangeX[0] - rangeX[1]) / d)  # minimum number of control points needed to avoid aliasing
# Avoiding spatial aliasing
while cps < cps_min:
    cps = cps + 1

x = np.linspace(rangeX_expanded[0], rangeX_expanded[1], cps)
yb = np.linspace(rangeY_B_expanded[0], rangeY_B_expanded[1], cps)
yd = np.linspace(rangeY_D_expanded[0], rangeY_D_expanded[1], cps)
y = np.concatenate((yd, yb))
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()

idx_cp = np.arange(0, len(point_lr), len(point_lr)/sample_cp, dtype=int)  # relative to listening area
point_cp = point_lr[idx_cp]
point_cp_temp = np.array([x, y, 2*np.ones(x.shape)]).transpose()

# selecting indexes of control points that are a subcategory of evaluation points
for n_pX in range(point_cp_temp.shape[0]):
    if (point_cp_temp[n_pX, 0] >= (rangeX_expanded[0])) & (point_cp_temp[n_pX, 0] <= (rangeX_expanded[1])):
        if (((point_cp_temp[n_pX, 1] >= (rangeY_B_expanded[0])) & (point_cp_temp[n_pX, 1] <= (rangeY_B_expanded[1]))) | ((point_cp_temp[n_pX, 1] <= (rangeY_D_expanded[0])) & (point_cp_temp[n_pX, 1] >= (rangeY_D_expanded[1])))):
            if first:
                point_cp = np.expand_dims(point_cp_temp[n_pX], axis=0)
                idx_cp = np.expand_dims(n_pX, axis=0)
                first = False
            else:
                point_cp = np.concatenate([point_cp, np.expand_dims(point_cp_temp[n_pX], axis=0)])
                idx_cp = np.concatenate([idx_cp, np.expand_dims(n_pX, axis=0)])

for n_p in range(len(point_cp)):
    idx_cp[n_p] = np.argmin(np.linalg.norm(point_cp[n_p] - point_lr, axis=1))

######### SEMICIRCULAR VIRTUAL SOURCES #########################

#step_radius = 0.2
#radius_sources_train = np.arange(1, 3, step_radius)
#radius_sources_test = np.arange(1 + (step_radius / 2), 3 + step_radius + (step_radius / 2), step_radius)
#n_sources_radius = 128
#src_pos_train = np.zeros((len(radius_sources_train) * n_sources_radius, 2))
#src_pos_test = np.zeros((len(radius_sources_train), n_sources_radius, 2))

#angles = np.linspace(3 * np.pi / 4, 5 * np.pi / 4, n_sources_radius)
#for n_r in range(len(radius_sources_train)):
    #for n_s in range(n_sources_radius):
        #src_pos_train[(n_r * n_sources_radius) + n_s] = sg.pol2cart(radius_sources_train[n_r], angles[n_s])
        #src_pos_test[n_r, n_s] = sg.pol2cart(radius_sources_test[n_r], angles[n_s])
###################################################################

######### VIRTUAL SOURCES #########################

idx_cp_x2 = idx_lr_gd_x[0:-1:2]  # 0.0416 [m] of spacing between mics -> aliasing @ 4086 [Hz]
idx_cp_x3 = idx_lr_gd_x[0:-1:3]  # 0.0625 [m] of spacing between mics -> aliasing @ 2720 [Hz]
idx_cp_x4 = idx_lr_gd_x[0:-1:4]  # 0.083 [m] of spacing between mics -> aliasing @ 2048 [Hz]
idx_cp_x5 = idx_lr_gd_x[0:-1:5]  # 0.1 [m] of spacing between mics -> aliasing @ 1700 [Hz]
idx_cp_x6 = idx_lr_gd_x[0:-1:6]  # 0.125 [m] of spacing between mics -> aliasing @ 1360 [Hz]

idx_cp_x2_expanded = idx_lr_gd_x_expanded[0:-1:2]
idx_cp_x3_expanded = idx_lr_gd_x_expanded[0:-1:3]
idx_cp_x4_expanded = idx_lr_gd_x_expanded[0:-1:4]
idx_cp_x5_expanded = idx_lr_gd_x_expanded[0:-1:5]
idx_cp_x6_expanded = idx_lr_gd_x_expanded[0:-1:6]




idx_cp_y2 = idx_lr_gd_y[0:-1:2]  # 0.0416 [m] of spacing between mics -> aliasing @ 4086 [Hz]
idx_cp_y3 = idx_lr_gd_y[0:-1:3]  # 0.0625 [m] of spacing between mics -> aliasing @ 2720 [Hz]
idx_cp_y4 = idx_lr_gd_y[0:-1:4]  # 0.083 [m] of spacing between mics -> aliasing @ 2048 [Hz]
idx_cp_y5 = idx_lr_gd_y[0:-1:5]  # 0.1 [m] of spacing between mics -> aliasing @ 1700 [Hz]
idx_cp_y6 = idx_lr_gd_y[0:-1:6]  # 0.125 [m] of spacing between mics -> aliasing @ 1360 [Hz]

idx_cp_y2_expanded = idx_lr_gd_y_expanded[0:-1:2]
idx_cp_y3_expanded = idx_lr_gd_y_expanded[0:-1:3]
idx_cp_y4_expanded = idx_lr_gd_y_expanded[0:-1:4]
idx_cp_y5_expanded = idx_lr_gd_y_expanded[0:-1:5]
idx_cp_y6_expanded = idx_lr_gd_y_expanded[0:-1:6]




# Setting ranges of training set
x_min_train = - 3.75
x_max_train = - 1.75
y_min_train = - 1.5
y_max_train = 1.5

# arbitrary spacing
sources_spacing_x = 0.04
sources_spacing_y = 0.1
virtual_s_x = int(abs(x_min_train - x_max_train) / sources_spacing_x)  # quantity of columns
virtual_s_y = int(abs(y_min_train - y_max_train) / sources_spacing_y)  # quantity of rows

# train positions
x_train = np.linspace(x_min_train, x_max_train, virtual_s_x)
y_train = np.linspace(y_min_train, y_max_train, virtual_s_y)
[X, Y] = np.meshgrid(x_train, y_train)
src_pos_train = np.array([X.ravel(), Y.ravel()])

# Separating test sources from training sources
train_test_dist_x = np.abs((x_min_train - x_max_train) / virtual_s_x) / 2
train_test_dist_y = np.abs((y_min_train - y_max_train) / virtual_s_y) / 2

# test positions
x_test = np.linspace(x_min_train + train_test_dist_x, x_max_train + train_test_dist_x, virtual_s_x)
y_test = np.linspace(y_min_train + train_test_dist_y, y_max_train + train_test_dist_y, virtual_s_y)
[X_test, Y_test] = np.meshgrid(x_test, y_test)
src_pos_test = np.array([X_test.ravel(), Y_test.ravel()])

# reduced grids to consider only control points zones in representation
grid_x2 = grid2D[0].T[idx_cp_x2]
grid_y2 = grid2D[0].T[idx_cp_y2]

grid_x3 = grid2D[0].T[idx_cp_x3]
grid_y3 = grid2D[0].T[idx_cp_y3]

grid_x4 = grid2D[0].T[idx_cp_x4]
grid_y4 = grid2D[0].T[idx_cp_y4]

plot_setup_1 = True
#plot_setup = False
if plot_setup_1:
    plt.figure(figsize=(10, 10))
        #plt.plot(point[:, 0], point[:, 1], 'r*')
    #plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
        #plt.plot(src_pos_train[:, 0], src_pos_train[:, 1], 'c*')
        #plt.plot(src_pos_test[:, :, 0], src_pos_test[:, :, 1], 'r*')
        #plt.plot(src_pos_test[:,  0], src_pos_test[:, 1], 'r*')
        #plt.plot(src_pos_train[0,:], src_pos_train[1,:],'c*')
        #plt.plot(src_pos_test[0,:], src_pos_test[1,:],'r*')
    #plt.plot(src_pos_train[0,:], src_pos_train[1,:],'c*')
    #plt.plot(src_pos_test[0,:], src_pos_test[1,:],'r*')
    plt.xlim(-2, 2)
    plt.ylim(-3.79, 2)
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['Eval Points', 'Control Points', 'LoudSpeakers', 'Train Sources', 'Test Sources'])
    plt.title("Setup 2D")
    plt.show()

plot_setup_1_3D = True
#plot_setup = False
if plot_setup_1_3D:
    plt.figure(figsize=(10, 10))
    plt.plot(point[:, 0], point[:, 1], point[:, 2], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], point_lr[:, 2], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], point_cp[:, 2], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], array_pos[:, 2], 'k*')
    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-2, 2)
    #ax.set_zlim(0, 4)
    plt.title("Setup 3D")
    plt.show()
#print(str(len(point_cp)) + ' control points')

################## Test sources #######################################################################################
# Generate sources
start_point = -1.5
end_point = 1.5
num_sources = 16
step = (start_point - end_point) / num_sources  # 0.1875
dist_sources = np.linspace(start_point, end_point, num_sources)
sources_positions_train = np.zeros((len(dist_sources) * num_sources, 2))  # (256, 2)
sources_positions_test = np.zeros((len(dist_sources),  num_sources, 2))  # (16, 16, 2)

plot_setup_2 = False
if plot_setup_2:
    plt.figure(figsize=(10, 10))
        #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
        #plt.plot(point[idx_lr[idx_cp], 0], point[idx_lr[idx_cp], 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.plot(src_pos_train[:, 0], src_pos_train[:, 1], 'c*')
        #plt.plot(src_pos_test[:, :, 0], src_pos_test[:, :, 1], 'r*')
    plt.plot(src_pos_test[:, 0], src_pos_test[:, 1], 'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)
        #plt.setzlim(0, 4)
    plt.legend(['Eval Points', 'Control Points', 'LoudSpeakers', 'Train Sources', 'Test Sources'])
    plt.title("Setup 2D")
    plt.show()

plot_setup_2_3D = False
#ax = plt.figure().add_subplot(projection='3d')
if plot_setup_2_3D:
        #plt.figure(figsize=(10, 10))
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
        #ax.plot(x, y, zs=0, zdir='z')
        #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], point_lr[:, 2], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], point_cp[:, 2], 'b*')
        #plt.plot(point[idx_lr[idx_cp], 0], point[idx_lr[idx_cp], 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], array_pos[:, 2], 'k*')
    #plt.plot(src_pos_train[:, 0], src_pos_train[:, 1], 'c*')
    #plt.plot(src_pos_test[:, :, 0], src_pos_test[:, :, 1], 'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['eval points', 'control points', 'loudspeakers', 'train sources', 'test sources'])
    #ax.set_xlim(-2, 2)
    #ax.set_ylim(-2, 2)
    #ax.set_zlim(0, 4)
    #ax.view_init(elev=2., azim=0, roll=0)
    plt.title("Setup 3D")
    plt.show()

# needed in other scripts
src_pos_trainT = src_pos_train.T
src_pos_testT = src_pos_test.T

src_pos_train_3D = np.asarray([src_pos_train[0], src_pos_train[1], 2*np.ones_like(src_pos_train[0])])
src_pos_test_3D = np.asarray([src_pos_test[0], src_pos_test[1], 2*np.ones_like(src_pos_test[0])])

xs_near = src_pos_train_3D.T[-1]
xs_far = src_pos_train_3D.T[1]
xs_low = src_pos_train_3D.T[-29]
xs_high = src_pos_train_3D.T[-29]



# USE FOR PLOTTING IF IT SEEMS SOMETHING IS OFF WITH SFS'S METHOD
#plt.figure(figsize=(10, 20))
#p_gt = np.reshape(np.real(P_gt[-1, :, 36]), (int(len(c_points_y)), int(len(c_points_x))))
#plt.imshow(p_gt)
#plt.show()
plot_test = True
if plot_test:
    d_point_2d, selection_point_2d, secondary_source_point_2d = sfs.fd.wfs.point_2d(wc[41],
                                                                             array.x,
                                                                             array.n,
                                                                             xs=src_pos_train_3D.T[0],
                                                                             c=c)
    p_low_near, p_low_near_idx = sound_field(d_point_2d, selection_point_2d, secondary_source_point_2d, array, idx_x=idx_lr_gd_x, idx_y=idx_lr_gd_y, grid=grid2D, tapering=False, DDD=False, norm=False, db=False, xs=src_pos_train_3D.T[0])
    plt.xlim(x_min_train, 2)
    #plt.legend(['virtual source'])#, 'PM'])
    #plt.annotate('virtual source', xy=(xs_low[0], xs_low[1]), xytext=(-3.25, 0.5), arrowprops=dict(facecolor='black', shrink=0.05),)
    #plt.annotate('secondary sources array', xy=(array.x[0][0], array.x[0][1]), xytext=(-0.25, -1.75), arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.title("Ground Truth Sound Field w/o Dark Zone")#, 'PM'])

    plt.show()

    d_point_2d, selection_point_2d, secondary_source_point_2d = sfs.fd.wfs.point_2d(wc[16],
                                                                             array.x,
                                                                             array.n,
                                                                             xs=xs_high,
                                                                             c=c)
    p_low_far, p_low_far_idx = sound_field(d_point_2d, selection_point_2d, secondary_source_point_2d, array, idx_x=idx_lr_gd_x, idx_y=idx_lr_gd_y, grid=grid2D, tapering=False, DDD=False, norm=False, db=False, xs=xs_high)
    plt.xlim(x_min_train, 2)
    plt.legend(['low-near'])#, 'PM'])
    plt.show()


    d_point_2d, selection_point_2d, secondary_source_point_2d = sfs.fd.wfs.point_2d(wc[16],
                                                                             array.x,
                                                                             array.n,
                                                                             xs=xs_low,
                                                                             c=c)
    p_high_near, p_high_near_idx = sound_field(d_point_2d, selection_point_2d, secondary_source_point_2d, array, idx_x=idx_lr_gd_x, idx_y=idx_lr_gd_y, grid=grid2D, tapering=False, DDD=False, norm=False, db=True, xs=xs_low)
    plt.xlim(x_min_train, 2)
    plt.legend(['high-far'])#, 'PM'])
    plt.show()


    d_point_2d, selection_point_2d, secondary_source_point_2d = sfs.fd.wfs.point_2d(wc[16],
                                                                             array.x,
                                                                             array.n,
                                                                             xs=xs_high,
                                                                             c=c)
    p_high_far, p_high_far_idx = sound_field(d_point_2d, selection_point_2d, secondary_source_point_2d, array, idx_x=idx_lr_gd_x, idx_y=idx_lr_gd_y, grid=grid2D, tapering=False, DDD=False, norm=False, db=True, xs=xs_high)
    plt.xlim(x_min_train, 2)
    plt.title("Virtual Point Source's Sound Field")#, 'PM'])

    plt.show()

print("Params_Linear_2D --># Ended")

