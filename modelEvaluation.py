import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import train
from data_lib import params_linear_2D
import sfs
import tqdm
import train
from sklearn.model_selection import train_test_split


#for i in range(63):
    #plt.figure(figsize=(10, 20))
    #plt.subplot(311)
    #sfs.plot2d.amplitude(p_gt * normalization_point, mingrid)
    #sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
    #plt.title('GT, Point Source at {} m and frequency {}'.format(xs, params_linear_2D.f_axis[n_f]))
    #plt.subplot(312)
    #plt.plot(d_hat[i, :int(d_hat.shape[1] / 2), n_f])
    #plt.plot(d_hat[i, int(d_hat.shape[1] / 2):, n_f])
    #plt.legend(['real', 'imag'])
    #plt.title('est')
    #plt.subplot(313)
    #sfs.plot2d.amplitude(p_est_reshaped * normalization_point, mingrid)
    #sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
    #plt.title('est')
    #plt.show()

os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#
os.environ['CUDA_ALLOW_GROWTH']='True'
GRID = sfs.util.xyz_grid([-2, 2], [-2, 2], 2, spacing=0.02)
import scipy



def plot_soundfield(cmap, P, n_f, selection, axis_label_size, tick_font_size, save_path = None, plot_ldspks=True, do_norm=True,  grid=GRID):
    figure = plt.figure(figsize=(20, 20))
    if do_norm:
        im = sfs.plot2d.amplitude(np.reshape(P[:256, n_f], (params_linear_2D.N_sample, params_linear_2D.N_sample)),
                                  params_linear_2D.grid2D, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    else:
        im = sfs.plot2d.amplitude(np.reshape(P[:256, n_f], (params_linear_2D.N_sample, params_linear_2D.N_sample)),
                                  params_linear_2D.grid2D,  cmap=cmap, colorbar=False, vmin=P[:, n_f].min(), vmax=P[:, n_f].max(), xnorm=None)
    if plot_ldspks:
        sfs.plot2d.loudspeakers(params_linear_2D.array.x[selection], params_linear_2D.array.n[selection], a0=1, size=0.18)
    plt.xlabel('$x [m]$', fontsize=axis_label_size), plt.ylabel('$y [m]$', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    cbar = plt.colorbar(im, fraction=0.046)
    cbar.ax.tick_params(labelsize=tick_font_size)
    # cbar.set_label('$NRE~[\mathrm{dB}]$',fontsize=tick_font_size))
    plt.tight_layout()
    #plt.savefig(save_path)
    plt.show()

def sound_field(d, selection, secondary_source, array, grid, xs, tapering=True, DDD=False, norm=True, db=False):
    if tapering:
        tapering_window = sfs.tapering.tukey(selection, alpha=0.3)
        plt.plot(tapering_window)
        plt.axis()
    else:
        tapering_window = sfs.tapering.none(selection)
    # Compute sound field for a generic driving function
    # INPUT = driving function, weights, secondary source distribution, secondary source function
    # OUTPUT = Sound Field for a generic d (sound pressure values, ignores imaginary part)
    p = sfs.fd.synthesize(d, tapering_window, array, secondary_source, grid=grid)
    print("d, tapering_window, secondary_source, grid = {}, {}, {}, {}\np = {}".format(np.shape(d), np.shape(tapering_window), np.shape(secondary_source), np.shape(grid), np.shape(p)))

    if (DDD):
        sfs.plot3d.secondary_sources(p, grid[0], [0, 0, 0])
        sfs.plot3d.secondary_sources(array.x, array.n, tapering_window)
    else:
        if norm:
            if db:
                im_db = sfs.plot2d.level(p, grid, xnorm=[0, 0, 0])  # the normalization is respect to the point
                sfs.plot2d.add_colorbar(im_db)
            else:
                im = sfs.plot2d.amplitude(p, grid, xnorm=[0, 0, 0])  # the normalization is respect to the point
                sfs.plot2d.add_colorbar(im)

        else:
            if db:
                im_db = sfs.plot2d.level(p, grid)
                sfs.plot2d.add_colorbar(im_db)
            else:
                im = sfs.plot2d.amplitude(p, grid)
                sfs.plot2d.add_colorbar(im)

        sfs.plot2d.loudspeakers(array.x, array.n, tapering_window)
        #sfs.plot2d.secondary_sources(p, grid)  # check parameters
        sfs.plot2d.virtualsource(xs)

    return p


PLOT = True

n_lspk = params_linear_2D.N_lspks
idxy = params_linear_2D.idx_lr_gd_y
idx_y = idxy[int(len(idxy)/2):]

c_points_x = params_linear_2D.idx_cp_x2
c_points_y = params_linear_2D.idx_cp_y2
cp = len(c_points_x) * len(c_points_y)

lr = 0.001
lambda_abs = 25
lambda_D = 1/25 #np.finfo(dtype=np.float64).eps


test_semplice = False
no_weights = False
normed = False
control_points = True
eval_points = False

if test_semplice:
    model = tf.keras.models.load_model(
        '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_16easy')
elif no_weights:
    model = tf.keras.models.load_model(
        '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_16_no_weight')
elif normed:
    model = tf.keras.models.load_model(
        '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_16_normed')
elif control_points:
    #model = tf.keras.models.load_model(
        #'#/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_'+'_nl'+str(
            #params_linear_2D.N_lspks)+'_cp_'+str(cp)+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_y+1'+'_B'+str(1/lambda_D)+'D')   #'_lr'+str(lr)+'_y+1')
    model = tf.keras.models.load_model('/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_64_cp_300_lambda1_lr0.001_y+1_B24.99999999999986D')

elif eval_points and not test_semplice:
    model = tf.keras.models.load_model(
        '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_16')



#model = tf.keras.models.load_model('model_linear_config_nl_64_cp_300_lambda1_lr0.001_y+1_B24.99999999999986D')
# It should be already compiled !

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                                                                # /test/green...
green_function_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/green_function_sec_sources_'+'_nl'+str(
            params_linear_2D.N_lspks)+ '_r_-0.25.npy'


if control_points:
    gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train'+'_nl'+str(
            params_linear_2D.N_lspks)+'_half_cp_double_train_0.npy'
elif eval_points:
    gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train_eval_points.npy'
else:
    gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train.npy'

#green_function_path = '/nas/home/ralessandri/thesis_project/dataset/test/green_function_sec_sources_nl_' + str(n_lspk) + '_r_-0.25.npy'
#gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/test/gt_soundfield_test.npy'


if control_points:
    sf_shape_x = int(
    len(c_points_x) * 2)  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
    sf_shape_y = int(
    len(c_points_y) * 2 / 2)  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
    sf_shape = sf_shape_x * sf_shape_y
elif eval_points:
    sf_shape_x = int(
    len(params_linear_2D.idx_lr_gd_x) * 2)  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
    sf_shape_y = int(
    len(params_linear_2D.idx_lr_gd_y) * 2 / 2)  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
    sf_shape = sf_shape_x * sf_shape_y

G = np.load(green_function_path)  # (16384, 16, 64) - (points, speakers, frequencies)

if control_points:
    P_gt_ = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(c_points_x)),
                      int(len(c_points_y)), params_linear_2D.N_freqs), dtype=complex)
elif eval_points:
    P_gt_ = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(params_linear_2D.idx_lr_gd_x)),
                  int(len(params_linear_2D.idx_lr_gd_y)), params_linear_2D.N_freqs), dtype=complex)
print("shape G = {}".format(np.shape(G)))
# print("loaded error shape = {}".format(np.shape(np.load(gt_soundfield_dataset_path))))
P_gt_ = np.load(gt_soundfield_dataset_path)  # gt soundfield
print("shape P_gt = {}".format(np.shape(P_gt_)))
if control_points:
    P_gt = np.zeros((len(params_linear_2D.src_pos_trainT),
                     int(len(c_points_x)) * int(len(c_points_y)),
                     params_linear_2D.N_freqs), dtype=complex)
elif eval_points:
    P_gt = np.zeros((len(params_linear_2D.src_pos_trainT),
                 int(len(params_linear_2D.idx_lr_gd_x)) * int(len(params_linear_2D.idx_lr_gd_y)),
                 params_linear_2D.N_freqs), dtype=complex)

#P_gt = np.load(gt_soundfield_dataset_path)  # gt soundfield, (675, 166, 64) - (sources, points, frequencies)
#G_cp = G[params_linear_2D.idx_lr[params_linear_2D.idx_cp]]  # (166, 16, 64) - (points, speakers, frequencies)
if control_points:
    G_cp_y = G[c_points_y]
    G_cp_ = G_cp_y[:, c_points_x]
elif eval_points:
    G_cp_y = G[params_linear_2D.idx_lr_gd_y]
    G_cp_ = G_cp_y[:, params_linear_2D.idx_lr_gd_x]
print("np.shape(G_cp_) = {}".format(np.shape(G_cp_)))
G_cp = np.zeros((int(sf_shape / 2), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)


for i in range(params_linear_2D.N_lspks):
    for j in range(params_linear_2D.N_freqs):
        G_cp[:, i, j] = np.ravel(G_cp_[:, :, i, j])

G = tf.convert_to_tensor(G)  # (16384, 16, 64) - (points, speakers, frequencies)
G_cp = tf.convert_to_tensor(G_cp)  # (166, 16, 64) - (points, speakers, frequencies)


val_perc = 0.01

def concat_real_imag(P_, src):
    P_concat = tf.concat([tf.math.real(P_), tf.math.imag(P_)], axis=0)
    return P_concat, P_, src

def preprocess_dataset(P, src):
    data_ds = tf.data.Dataset.from_tensor_slices((P, src))
    preprocessed_ds = data_ds.map(concat_real_imag)
    return preprocessed_ds

test_ds = preprocess_dataset(P_gt, params_linear_2D.src_pos_test.T)

i = 1
test_ds_ = []
P_concat = test_ds

selection = np.ones_like(params_linear_2D.array_pos[:, 0])

#plt.figure(figsize=(16,5))
#plt.subplot(1,2,1)
## summarize history for accuracy
#plt.plot(model.history['accuracy'])
#plt.plot(model.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')

#plt.subplot(1,2,2)
## summarize history for loss
#plt.plot(model.history['loss'])
#plt.plot(model.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'val'], loc='upper left')



if eval_points:
    N_pts = len(params_linear_2D.idx_lr)
elif control_points:
    N_pts = len(c_points_x)*len(c_points_y)
else:
    N_pts = len(params_linear_2D.point)


x0 = -2, 1, 2
f = 500  # Hz
omega = 2 * np.pi * f
normalization_point = 4 * np.pi
normalization_line = np.sqrt(8 * np.pi * omega / sfs.default.c) * np.exp(1j * np.pi / 4)
grid = params_linear_2D.grid2D
vgrid = sfs.util.xyz_grid([-2, 2], [-2, 2], 2, spacing=0.1)  # grid for vector fields
p = sfs.fd.source.point(omega, x0, grid) * np.linalg.norm(grid - x0)   # exp(-j*wc/c*np.linalg.norm(grid - x0))/4pi
print("p in modeleval = {}".format(np.shape(p)))

if control_points:
    grid_x = grid[0].T[c_points_x]
    grid_y = grid[1][c_points_y]
elif eval_points:
    grid_x = grid[0].T[params_linear_2D.idx_lr_gd_x]
    grid_y = grid[1][params_linear_2D.idx_lr_gd_y]

print("grid_x, grid_x0, grid_x00 = {}, {}, {}".format(np.shape(grid_x), np.shape(grid_x[0]), np.shape(grid_x[0][0])))
print("grid_y, grid_y0, grid_y00 = {}, {}, {}".format(np.shape(grid_y), np.shape(grid_y[0]), np.shape(grid_y[0][0])))
print("grid, grid[0], grid[1] = {}, {}, {}".format(np.shape(grid), np.shape(grid[0]), np.shape(grid[1])))

mingrid = sfs.util.xyz_grid([-0.25, 0.25], [-0.5, 0.5], 2, spacing=0.02)

for n_r in tqdm.tqdm(range(len(params_linear_2D.src_pos_trainT))):


    for n_f in range(len(params_linear_2D.wc)):
        N_pts = 201  # len(params_linear_2D.grid2D)

        point = params_linear_2D.point

        if control_points:
            P_into = np.zeros((int(len(c_points_x)), int(len(c_points_y))), dtype=complex)
        elif eval_points:
            P_into = np.zeros((int(len(params_linear_2D.idx_lr_gd_x)),
                           int(len(params_linear_2D.idx_lr_gd_y))), dtype=complex)

        if test_semplice:
            P_into = P_gt_[-1, :, :, n_f]  * normalization_point
            xs = np.append(params_linear_2D.src_pos_trainT[-1], 2)
        else:
            P_into = P_gt_[-(n_r + 1), :, :, n_f] #* normalization_point -!
            G_cp_into = G_cp_[:, :, -(n_r + 1), n_f]
            xs = np.append(params_linear_2D.src_pos_trainT[-(n_r+1)], 2)

        P_gt[n_r, :, n_f] = np.ravel(P_into)


        P_gt_source = P_gt[n_r, :, :]


        P_pwd_cnn = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_pm = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        print("xs = {} with type {}, while x0 = {} with type {}".format(xs, type(xs), x0, type(x0)))

        #d_array_cnn = model.predict(np.expand_dims(np.concatenate([np.real(P_gt_source), np.imag(P_gt_source)], axis=0), axis=[0, -1]).astype('float32'))[0, :, :, 0].astype('float64')
        d_array_cnn = model(np.expand_dims(np.concatenate([np.real(P_gt_source), np.imag(P_gt_source)], axis=0), axis=[0, -1]).astype('float32'), training=False)[0, :, :, 0].astype('float64')  # 32, 64, 64 <-- Sources, CPs, Freqs

        d_array_cnn_complex = d_array_cnn[:int(d_array_cnn.shape[0] / 2)] + (1j * d_array_cnn[int(d_array_cnn.shape[0] / 2):])


        #plt.subplot(3, 1, 1)
        #plt.plot(np.real(d_array_cnn_complex.T[10, :]))
        #plt.subplot(3, 1, 2)
        #plt.plot(np.imag(d_array_cnn_complex.T[10, :]))
        #plt.subplot(3, 1, 3)
        #plt.plot(np.imag(d_array_cnn_complex.T[10, :]))
        #plt.show()


        #plt.figure(figsize=(10, 10))
        #plt.imshow(d_array_cnn.T, aspect='auto', cmap='RdBu')
        #plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
        #plt.tick_params(axis='both', which='major', labelsize=120)
        #plt.gca().invert_yaxis()
        #plt.title("d_array_cnn")
        #plt.show()


        if PLOT:

            # Plot params
            #selection = np.ones_like(params_linear_2D.array_pos[:, 0])
            #selection = selection == 1
            #n_f = 63  # 63
            #print("freq = ",str(params_linear_2D.f_axis[n_f]))
            #cmap = 'RdBu_r'
            #tick_font_size = 70
            #axis_label_size = 908
                                                                                                                        # params_linear_2D.npw if plane wave instead of xs
            d_line_2d, selection_point_2d, secondary_source_point_2d = sfs.fd.wfs.point_2d(params_linear_2D.wc[10], params_linear_2D.array.x, params_linear_2D.array.n, [xs.T[0], xs.T[1], 2])


            #plt.figure(figsize=(10, 10))
            ##plt.imshow(np.real(d_line_2d), aspect='auto', cmap='RdBu')
            #plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
            #plt.tick_params(axis='both', which='major', labelsize=120)
            #plt.gca().invert_yaxis()
            #plt.title("d_line_2d")
            #plt.show()

            print("d_complex, d_hat = {}, {}".format(np.shape(d_array_cnn_complex), np.shape(d_line_2d)))
            #p_est = tf.einsum('bij,kij->bkj', d_array_cnn_complex[:len(G_cp[0]), :], G_cp)
            ##    32 CPs, 64 Freqs    VS    166 Points, 16 Speakers, 64 Freqs    ##
            print("d, G = {}, {}".format(np.shape(d_array_cnn_complex), np.shape(G_cp)))
            p_est = tf.einsum('ij,kij->kj', d_array_cnn_complex, G_cp)
            print("p_est = {}".format(np.shape(p_est)))
            if control_points:
                p_est_reshaped = np.reshape(p_est[:, n_f], (int(len(c_points_y)), int(len(c_points_x))))
            elif eval_points:
                p_est_reshaped = np.reshape(p_est[:, n_f], (int(len(params_linear_2D.idx_lr_gd_y)), int(len(params_linear_2D.idx_lr_gd_x))))

            # Ground truth
            plot_paths = os.path.join('plots', 'linear')
            save_path = os.path.join(plot_paths, 'sf_real_source_' + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(n_lspk) + '.pdf')
            #plot_soundfield(cmap, P_gt_source, n_f, selection, axis_label_size, tick_font_size, save_path, plot_ldspks=False)
            #sound_field(d_line_2d, selection_line_2d, secondary_source_line_2d, params_linear_2D.array, grid=params_linear_2D.grid2D, tapering=False, DDD=False, norm=False, xs=xs)

            plt.figure(figsize=(9, 16))
            plt.subplot(211)
            #sfs.plot2d.amplitude(P_into_ * normalization_point, mingrid, colorbar_kwargs=dict(label="p / Pa"))
            sfs.plot2d.amplitude(P_into * normalization_point, mingrid, colorbar_kwargs=dict(label="p / Pa"))
            sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
            # Scommenta per fare test su specifiche frequenze
            #P__ = P_gt[n_r, :, :, 25]
            #P__y = P__[params_linear_2D.idx_lr_gd_y]
            #P___ = P__y[:, params_linear_2D.idx_lr_gd_x]
            #sfs.plot2d.amplitude(P___ * normalization_point, mingrid, colorbar_kwargs=dict(label="p / Pa"))
            plt.title("Point Source at {} m and frequency {} (normalized) - INPUT".format(xs, params_linear_2D.f_axis[n_f]))
            #plt.legend(['GT'])  # , 'PM'])

            lambda_normalizator = np.abs(np.mean(P_into)) / np.abs(np.mean(p_est_reshaped))
            plt.subplot(212)
            im = sfs.plot2d.amplitude(np.array(p_est_reshaped) * normalization_point, mingrid, colorbar_kwargs=dict(label="p / Pa"))  # the normalization is respect to the point
            #im = sfs.plot2d.amplitude(G_cp_into * normalization_point, mingrid, colorbar_kwargs=dict(label="p / Pa"))  # the normalization is respect to the point
            sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
            sfs.plot2d.add_colorbar(im)
            plt.title("DNN Sound Field")
            plt.show()

            #sound_field(d_line_2d, selection_line_2d, secondary_source_line_2d, params_linear_2D.array, grid=params_linear_2D.grid2D, tapering=True, DDD=False, norm=False, xs=xs)
            #sound_field(d_line_2d, selection_line_2d, secondary_source_line_2d, params_linear_2D.array, grid=params_linear_2D.grid2D, tapering=True, DDD=False, norm=True, xs=xs)
            #sound_field(d_line_2d, selection_line_2d, secondary_source_line_2d, params_linear_2D.array, grid=params_linear_2D.grid2D, tapering=True, DDD=False, norm=True, db=True, xs=xs)
            #plt.legend(['GT'])#, 'PM'])
            #plt.show()


            # PWD-CNN
            save_path = os.path.join(plot_paths,'sf_pwd_cnn_' + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(n_lspk) + '.pdf')

            #sound_field(p_est, selection, secondary_source_point_2d, array=params_linear_2D.array,
                        #grid=params_linear_2D.grid2D, tapering=True, DDD=False, xs=xs)
            #plt.legend(['PWD-CNN, SELECTION'])  # , 'PM'])
            #plt.show()
            #sound_field(p_est, selection_point_2d, secondary_source_point_2d, array=params_linear_2D.array,
                        #grid=params_linear_2D.grid2D, tapering=False, DDD=False, xs=xs)
            #plt.legend(['PWD-CNN, SELECTION-point'])  # , 'PM'])
            #plt.show()
            print("selection, selection_point 2d = {}, {}".format(selection, selection_point_2d))


            # PM
            #save_path = os.path.join(plot_paths,+'sf_pm_' + str(n_s) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(+n_lspk) + '.pdf')
            #plot_soundfield(cmap, P_pwd_pm, n_f, selection, axis_label_size, tick_font_size, save_path)

            #plt.legend(['DNN SOUND FIELD'])#, 'PM'])




P_test_ = np.expand_dims(P_gt, axis=0)

preds = model.predict(P_test_[:, :332, :64, :1])
src_test_ = np.expand_dims(params_linear_2D.src_pos_test, axis=-1)

plt.figure(figsize=(10, 20))
plt.scatter(P_gt[:332, :2, 0],src_test_[:332, :, 0])
plt.legend(['Test', 'Predicted']);
plt.show

model.compile(optimizer=optimizer, metrics=['accuracy'])
model.evaluate(P_test_[:, :332, :64, :1], params_linear_2D.src_pos_test.T[:1, :])

figure_soundfield = plt.figure(figsize=(10, 20))
#plt.subplot(211)
sfs.plot2d.amplitude(10*np.log10(P_gt[:332, :64, :1]), params_linear_2D.grid2D, xnorm=[0, 0, 0])
sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
plt.title('GT')

#plt.subplot(212)
#sfs.plot2d.amplitude(train.p_hat_real_, params_linear_2D.grid2D, xnorm=[0, 0, 0])
#sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
#plt.title('est')

# plt.show()

# figure_soundfield = train_utils.est_vs_gt_soundfield(tf.expand_dims(P_hat_real[:, :, :, idx_plot], axis=3), tf.expand_dims(P_real[:, :, :, idx_plot], axis=3))

#preds = model.predict(X_train)
#plt.scatter(X_train,y_train)
#plt.scatter(X_train,preds,c='r');
#plt.legend(['Train', 'Predicted']);
#model.evaluate(X_train, y_train)

#preds = model.predict(test_ds)
#plt.scatter(test_ds,src_test)
#plt.scatter(test_ds,preds,c='r');
#plt.legend(['Train', 'Predicted']);
#model.evaluate(test_ds, src_test)

plot_setup_1 = True
#plot_setup = False
if plot_setup_1:
    plt.figure(figsize=(10, 10))
    #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(P_gt[:332, :2, 0],src_test_[:332, :, 0], 'g*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['P_test'])
    plt.title("P_test")
    plt.show()

if PLOT:
    plt.figure(), plt.plot(params_linear_2D.array_pos[:4, 0], params_linear_2D.array_pos[:4, 1], 'r*'), plt.show()
    d = np.linalg.norm(np.array([params_linear_2D.array_pos[1, 0], params_linear_2D.array_pos[1, 1]]) - np.array(
        [params_linear_2D.array_pos[2, 0], params_linear_2D.array_pos[2, 1]]))
    aliasing_freq = params_linear_2D.c / (2 * d)
    # Plot params
    selection = np.ones_like(params_linear_2D.array_pos[:, 0])
    selection = selection == 1
    n_f = 63  # 63
    print(str(params_linear_2D.f_axis[n_f]))
    cmap = 'RdBu_r'
    tick_font_size = 70
    axis_label_size = 90

    # Ground truth
    plot_paths = os.path.join('plots', 'linear')
    #save_path = os.path.join(plot_paths, 'sf_real_source_' + str(n_s) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(n_lspk) + '.pdf')
    #plot_soundfield(cmap, P_gt, n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=False) <-- !!!! this should be UNCOMMENTED

    # PWD-CNN
    #save_path = os.path.join(plot_paths,'sf_pwd_cnn_' + str(n_s) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(n_lspk) + '.pdf')
    #plot_soundfield(cmap, P_pwd_cnn, n_f, selection, axis_label_size, tick_font_size, save_path)

    # PM
    #save_path = os.path.join(plot_paths, 'sf_pm_' + str(n_s) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(args.n_loudspeakers) + '.pdf')
    #plot_soundfield(cmap, P_pwd_pm, n_f, selection, axis_label_size, tick_font_size, save_path)

    # Error
    # nmse_pm = 10*np.log10(nmse(P_pwd_pm, P_gt, type='full'))
    # save_path = os.path.join(plot_paths, 'nmse_pm_' + str(n_s) + '_f_' + str(params_linear_2D.f_axis[n_f]) +'_nl'+str(args.n_loudspeakers)+ '.pdf')
    # plot_soundfield(cmap, nmse_pm, n_f, selection, axis_label_size, tick_font_size, save_path, do_norm=False)

    plt.figure()
    # plt.plot(params_linear_2D.f_axis, 10*np.log10(np.mean(nmse(P_pwd_cnn, P_gt, type='full'), axis=0)),'k-*')
    # plt.plot(params_linear_2D.f_axis, 10*np.log10(np.mean(nmse(P_pwd_pm, P_gt, type='full'), axis=0)),'r-*')
    plt.tick_params(axis='both', which='major', labelsize=10)

    #plt.legend(['CNN', 'PM'])
    plt.legend(['GT'])

    plt.show()

    print('pause')

print("Ended Evaluation")
