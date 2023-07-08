import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_lib import params_linear_2D
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

print("started plotting")

expanded = True
switched = False
cp =len(params_linear_2D.idx_cp_x2_expanded * 2)
if expanded:
    print("\nEXPANDED\n")
    gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train' + '_nl' + str(
        params_linear_2D.N_lspks) + '_half_cp'+str(cp)+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'
    gt_soundfield_dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
        params_linear_2D.N_lspks) + '_half_cp'+str(cp)+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'
    gt_soundfield_dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train' + '_nl' + str(
        params_linear_2D.N_lspks) + '_half_cp'+str(cp)+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'

    green_function_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/green_function_sec_sources' + '_nl_' + str(
        params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy'
    green_function_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/green_function_sec_sources' + '_nl_' + str(
        params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy'
    green_function_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/green_function_sec_sources' + '_nl_' + str(
        params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy'
else:
    if switched:
        print("\nSWITCHED\n")
        gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp_double_train_0_y+1' + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay_switch.npy'
        gt_soundfield_dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp_double_train_0_y+1' + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay_switch.npy'
        gt_soundfield_dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp_double_train_0_y+1' + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay_switch.npy'
    else:
        gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp_double_train_0_y+1' + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay.npy'
        gt_soundfield_dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp_double_train_0_y+1' + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay.npy'
        gt_soundfield_dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp_double_train_0_y+1' + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay.npy'

    green_function_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/green_function_sec_sources' + '_nl_' + str(
        params_linear_2D.N_lspks) + '_r_-0.25_decay.npy'
    green_function_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/green_function_sec_sources' + '_nl_' + str(
        params_linear_2D.N_lspks) + '_r_-0.25_decay.npy'
    green_function_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/green_function_sec_sources' + '_nl_' + str(
        params_linear_2D.N_lspks) + '_r_-0.25_decay.npy'

G = np.load(green_function_path)  # green function
G_B = np.load(green_function_path_bright)  # green function bright zone
G_D = np.load(green_function_path_dark)  # green function dark zone


if expanded:
    c_points_x = params_linear_2D.idx_cp_x2_expanded
    c_points_y = params_linear_2D.idx_cp_y2_expanded
else:
    c_points_x = params_linear_2D.idx_cp_x2
    c_points_y = params_linear_2D.idx_cp_y2

c_pointsx_y = c_points_y[int(len(c_points_y) / 2):]

G_cp = np.zeros(
    ( int(len(c_points_y)), int(len(c_points_x)), params_linear_2D.N_lspks, params_linear_2D.N_freqs),
    dtype=complex)
G_cp_B = np.zeros(
    ( int(len(c_pointsx_y)), int(len(c_pointsx_y)), params_linear_2D.N_lspks, params_linear_2D.N_freqs),
    dtype=complex)
G_cp_D = np.zeros(
    ( int(len(c_pointsx_y)), int(len(c_pointsx_y)), params_linear_2D.N_lspks, params_linear_2D.N_freqs),
    dtype=complex)

for n_s in range(G_cp.shape[2]):
    for n_f in range(G_cp.shape[3]):
        into_G_cp = G[:, :, n_s, n_f]
        into_G_cp_y = into_G_cp[c_points_y]
        into_G_cp_B = G_B[0:-1:2, 0:-1:2, n_s, n_f]
        into_G_cp_D = G_D[0:-1:2, 0:-1:2, n_s, n_f]

        G_cp[:, :, n_s, n_f] = into_G_cp_y[:, c_points_x]
        G_cp_B[:, :, n_s, n_f] = into_G_cp_B
        G_cp_D[:, :, n_s, n_f] = into_G_cp_D


P_gt_ = np.load(gt_soundfield_dataset_path)  # gt soundfield
P_gt__B = np.load(gt_soundfield_dataset_path_bright)  # gt soundfield bright zone
P_gt__D = np.load(gt_soundfield_dataset_path_dark)  # gt soundfield dark zon

check_plots = True
freq=41
if check_plots:
    plt.figure(figsize=(10, 20))
    plt.subplot(311)
    plt.imshow(np.real(G[:, :, -1, freq]))
    plt.subplot(312)
    plt.imshow(np.real(G_B[:, :, -1, freq]))
    plt.subplot(313)
    plt.imshow(np.real(G_D[:, :, -1, freq]))
    plt.show()

    plt.figure(figsize=(10, 20))
    plt.subplot(311)
    plt.imshow(np.real(P_gt_[-1, :, :, freq]))
    plt.subplot(312)
    plt.imshow(np.real(P_gt__B[-1, :, :, freq]))
    plt.subplot(313)
    plt.imshow(np.real(P_gt__D[-1, :, :, freq]))
    plt.show()

P_gt = np.zeros((P_gt_.shape[0], P_gt_.shape[1] * P_gt_.shape[2], P_gt_.shape[3]), dtype=np.complex64)

if expanded:
    mingrid_B = params_linear_2D.mingrid_B_expanded
    mingrid_D = params_linear_2D.mingrid_D_expanded
    #mingrid = params_linear_2D.mingrid_expanded
else:
    mingrid_B = params_linear_2D.mingrid_B
    mingrid_D = params_linear_2D.mingrid_D
    mingrid = params_linear_2D.mingrid

P_gt_B = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(mingrid_B[0][0]) / 2) * int(len(mingrid_B[1]) / 2),
                   params_linear_2D.N_freqs), dtype=complex)
P_gt_D = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(mingrid_D[0][0]) / 2) * int(len(mingrid_D[1]) / 2),
                   params_linear_2D.N_freqs), dtype=complex)

for i in range(len(params_linear_2D.src_pos_trainT)):
    for j in range(params_linear_2D.N_freqs):
        # P_to_ravel = P_gt_[i, :, :, j]
        P_to_ravel_B = P_gt__B[i, 0:-1:2, 0:-1:2, j]
        P_to_ravel_D = P_gt__D[i, 0:-1:2, 0:-1:2, j]

        # P_gt[i, :, j] = np.ravel(P_to_ravel)
        P_gt_B[i, :, j] = np.ravel(P_to_ravel_B)
        P_gt_D[i, :, j] = np.ravel(P_to_ravel_D)

#plt.figure(figsize=(10, 20))
#plt.imshow(np.real(P_gt_[-1, :, :, 16]))
#plt.show()

sf_shape_x = int(len(c_points_x))  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
sf_shape_y = int(len(c_points_y))  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
sf_shape = sf_shape_x * sf_shape_y
sf_shape_ = int(len(c_pointsx_y)) ** 2

G_cp_ = np.zeros((int(sf_shape), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
G_cp__B = np.zeros((int(sf_shape_), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
G_cp__D = np.zeros((int(sf_shape_), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
G_r = np.zeros((int(G.shape[0]**2), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)


# (G_D.H * G_D + Identity).I * G_B.H * G_B
#(np.matrix(G_cp_[:, :, 16]).H * np.matrix(G_cp_[:, :, 16])) ** (-1) + np.matrix(G_cp_[:, :, 16]).H * np.matrix(P_gt_[-1, :, :, 16])

for i in range(params_linear_2D.N_lspks):
    for j in range(params_linear_2D.N_freqs):
        into_G_cp = G_cp[:, :, i, j]
        into_G_cp__B = G_cp_B[:, :, i, j]
        into_G_cp__D = G_cp_D[:, :, i, j]
        into_G_r = G[:, :, i, j]

        G_cp_[:, i, j] = np.ravel(into_G_cp)
        G_cp__B[:, i, j] = np.ravel(into_G_cp__B)
        G_cp__D[:, i, j] = np.ravel(into_G_cp__D)
        G_r[:, i, j] = np.ravel(into_G_r)

n_lspk = params_linear_2D.N_lspks
if expanded:
    idxy = params_linear_2D.idx_lr_gd_y_expanded
else:
    idxy = params_linear_2D.idx_lr_gd_y
idx_y = idxy[int(len(idxy)/2):]

lr = 0.003
lambda_abs = 10 + np.finfo(dtype=np.float16).eps
lambda_D = 1/lambda_abs + np.finfo(dtype=np.float16).eps


test_semplice = False
no_weights = False
normed = False
control_points = True
eval_points = False
if expanded:
    c_points_x = params_linear_2D.idx_cp_x2_expanded
    c_points_y = params_linear_2D.idx_cp_y2_expanded
else:
    c_points_x = params_linear_2D.idx_cp_x2
    c_points_y = params_linear_2D.idx_cp_y2

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
    if expanded:
        model = tf.keras.models.load_model('/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_64_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay_expanded_fs15reIm', compile=False)
    elif switched:
        model = tf.keras.models.load_model('/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_64_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay_switch')
    else:
        model = tf.keras.models.load_model('/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_64_cp_300_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'bn')
                                                                                                                   # nl64_xmin-6_cp338_lambda25_lr0.002_B0.9990243902439024D_decay
elif eval_points and not test_semplice:
    model = tf.keras.models.load_model(
        '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_16')

#model = tf.keras.models.load_model('model_linear_config_nl_64_cp_300_lambda1_lr0.001_y+1_B24.99999999999986D')
# It should be already compiled !
source = -1
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
d_array_cnn = model.predict(
    np.expand_dims(np.concatenate([np.real(P_gt_B[source, :, :]), np.imag(P_gt_B[source, :, :])], axis=0), axis=[0, -1]).astype('float32'))[0, :, :, 0].astype('float64')  # 32, 64, 64 <-- Sources, CPs, Freqs
d_array_cnn_complex = d_array_cnn[:int(d_array_cnn.shape[0] / 2)] + (1j * d_array_cnn[int(d_array_cnn.shape[0] / 2):])

p_est_cp = tf.einsum('ij,kij->kj', d_array_cnn_complex, G_cp_)
p_est = tf.einsum('ij,kij->kj', d_array_cnn_complex, G_r)
p_est_B = tf.einsum('ij,kij->kj', d_array_cnn_complex, G_cp__B)
p_est_D = tf.einsum('ij,kij->kj', d_array_cnn_complex, G_cp__D)

p_est_cp_reshaped = np.reshape(p_est_cp[:, freq], (int(len(c_points_x)), int(len(c_points_y))))
p_est_reshaped = np.reshape(p_est[:, freq], (int(G.shape[0]), int(G.shape[1])))
p_est_B_reshaped = np.reshape(p_est_B[:, freq], (int(len(c_points_x)), int(len(c_points_x))))
p_est_D_reshaped = np.reshape(p_est_D[:, freq], (int(len(c_points_x)), int(len(c_points_x))))

limit = int(np.max(np.abs(p_est_reshaped))) # / 12)
plt.figure(figsize=(10, 20))
plt.subplot(611)
plt.imshow(np.real(p_est_cp_reshaped))
plt.title("Pressure Matching at Control Points - DLPM")
plt.gca().invert_yaxis()

plt.subplot(612)
plt.imshow(4 * np.pi * np.real(p_est_reshaped))# / limit)
#plt.clim(-limit, limit)
plt.clim(-1, 1)
plt.colorbar()
plt.title("Pressure Matching at Complete Sound Field - DLPM, source = {}, lambda_abs = {}, lambda_dark = {}".format(source, lambda_abs, lambda_D))
plt.gca().invert_yaxis()

plt.subplot(613)
plt.imshow(np.real(p_est_B_reshaped))# / limit)
#plt.clim(-limit, limit)
#plt.clim(-1, 1)
plt.colorbar()
plt.title("Pressure Matching at Bright Sound Field - DLPM, source = {}, lambda_abs = {}, lambda_dark = {}".format(source, lambda_abs, lambda_D))
plt.gca().invert_yaxis()

plt.subplot(614)
plt.imshow(np.real(p_est_D_reshaped))# / limit)
#plt.clim(-limit, limit)
#plt.clim(-1, 1)
plt.colorbar()
plt.title("Pressure Matching at Dar Sound Field - DLPM, source = {}, lambda_abs = {}, lambda_dark = {}".format(source, lambda_abs, lambda_D))
plt.gca().invert_yaxis()

plt.subplot(615)
plt.plot(np.real(d_array_cnn_complex[:, freq]))# / limit)
#plt.clim(-limit, limit)
#plt.clim(-1, 1)
plt.title("Real part driving function")
plt.subplot(616)
plt.plot(np.imag(d_array_cnn_complex[:, freq]))# / limit)
#plt.clim(-limit, limit)
#plt.clim(-1, 1)
plt.title("Imaginary part driving function")

plt.show()

print("END DLPM")
print("")