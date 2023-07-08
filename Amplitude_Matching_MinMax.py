import numpy as np
import matplotlib.pyplot as plt
from data_lib import params_linear_2D
import tqdm
import os
#from numba import jit, cuda
import tensorflow as tf

os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['CUDA_ALLOW_GROWTH']='True'
cp =len(params_linear_2D.idx_cp_x2_expanded * 2)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
AUTOTUNE = tf.data.experimental.AUTOTUNE

print("start AM")


gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train' + '_nl' + str(
    params_linear_2D.N_lspks) + '_half_cp'+str(cp)+ '_xmin' + str(params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'
gt_soundfield_dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
    params_linear_2D.N_lspks) + '_half_cp'+str(cp)+ '_xmin' + str(params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'
gt_soundfield_dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train' + '_nl' + str(
    params_linear_2D.N_lspks) + '_half_cp'+str(cp)+ '_xmin' + str(params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'

green_function_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/green_function_sec_sources' + '_nl_' + str(
    params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy'
green_function_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/green_function_sec_sources' + '_nl_' + str(
    params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy'
green_function_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/green_function_sec_sources' + '_nl_' + str(
    params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy'

G = np.load(green_function_path)  # green function
G_B = np.load(green_function_path_bright)  # green function bright zone
G_D = np.load(green_function_path_dark)  # green function dark zone

expanded = True
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

for n_s in range(G_cp.shape[2]):
    for n_f in range(G_cp.shape[3]):
        into_G_cp = G[:, :, n_s, n_f]
        into_G_cp_y = into_G_cp[c_points_y]
        G_cp[:, :, n_s, n_f] = into_G_cp_y[:, c_points_x]

P_gt_ = np.load(gt_soundfield_dataset_path)  # gt soundfield
P_gt__B = np.load(gt_soundfield_dataset_path_bright)  # gt soundfield bright zone
P_gt__D = np.load(gt_soundfield_dataset_path_dark)  # gt soundfield dark zone

for i in range(P_gt_.shape[0]):
    for j in range(P_gt_.shape[-1]):
        P_gt_[i, :int(P_gt_.shape[1] / 2), :, j] = np.finfo(np.complex64).eps

#plt.figure(figsize=(10, 20))
#plt.imshow(np.real(P_gt_[-1, :, :, 16]))
#plt.show()
if expanded:
    mingrid_B = params_linear_2D.mingrid_B_expanded
    mingrid_D = params_linear_2D.mingrid_D_expanded
else:
    mingrid_B = params_linear_2D.mingrid_B
    mingrid_D = params_linear_2D.mingrid_D
mingrid = params_linear_2D.mingrid


sf_shape_x = int(len(c_points_x))  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
sf_shape_y = int(len(c_points_y))  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
sf_shape = sf_shape_x * sf_shape_y

G_cp_ = np.zeros((int(sf_shape), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
G_r = np.zeros((int(G.shape[0]**2), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)  # ?
# d_hat = (G^h * G + gammaI) + gamma d^h * desired
#(np.matrix(G_cp_[:, :, 16]).H * np.matrix(G_cp_[:, :, 16])) ** (-1) + np.matrix(G_cp_[:, :, 16]).H * np.matrix(P_gt_[-1, :, :, 16])

for i in range(params_linear_2D.N_lspks):
    for j in range(params_linear_2D.N_freqs):
        into_G_cp = G_cp[:, :, i, j]
        into_G_r = G[:, :, i, j]

        G_cp_[:, i, j] = np.ravel(into_G_cp)
        G_r[:, i, j] = np.ravel(into_G_r)

# ([speakers, points] * [points, speakers] + Identity).I * [speakers, points] * points
lambda_pm = 1e-3
d_hat_list_src = []
d_hat_pm = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_PM.npy')
max_iter=1000
dtol=1e-3
#@jit(target_backend='cuda')
def AM_MM_gpu(src, wc, G, lambdaPM, d_hatPM, P_gt, maxIter, dTol):
    print("\nusing GPU?\n")
    for n_src in tqdm.tqdm(range(len(src))):
        d_hat_list_f = []
        for n_f in tqdm.tqdm(range(len(wc))):

            A = ((np.matrix(G[:, :, n_f]).H * np.matrix(G[:, :, n_f])) + lambdaPM * np.identity((np.matrix(G[:, :, n_f]).H * np.matrix(G[:, :, n_f])).shape[0], dtype=np.complex64)).I * np.matrix(G[:, :, n_f]).H
            drv = d_hatPM[n_src, :, n_f, :]
            drvList = [drv]
            v = np.ravel(np.abs(P_gt[n_src, :, :, n_f])) * np.exp(1j * np.angle(G[:, :, n_f] @ drv)).T
            k = 0
            ddiff = 1.0
            for k in range(maxIter):
                drv = np.asarray(A) @ v.T
                drvList.append(drv)
                ddiff = np.linalg.norm(drvList[k + 1] - drvList[k]) / np.linalg.norm(drvList[k])
                v = np.ravel(np.abs(P_gt[n_src, :, :, n_f])) * np.exp(1j * np.angle(G[:, :, n_f] @ drv)).T
            # print("itr: %d, ddiff: %f" % (k, ddiff))
                if ddiff <= dTol:
                    break

            d_hat_list_f.append(drv)
        d_hat_list_src.append(d_hat_list_f)
    return np.asarray(d_hat_list_src)

srcT = params_linear_2D.src_pos_trainT[0:-1:10]
wc = params_linear_2D.wc
d_hats = AM_MM_gpu(srcT, wc, G_cp_, lambda_pm, d_hat_pm, P_gt_, max_iter, dtol)
save = True
if save:
    np.save(os.path.join('/nas/home/ralessandri/thesis_project/dataset', 'd_hat_AM150.npy'), d_hats)
    print('saved')

PLOT = False
if PLOT:
    p_est_cp = np.matrix(d_hat_list_src[29, 16, :, :].T) * G_cp_[:, :, 16].T
    p_est = np.matrix(d_hat_list_src[29, 16, :, :].T) * G_r[:, :, 16].T
    p_est_cp_reshaped = np.reshape(p_est_cp, (int(len(c_points_x)), int(len(c_points_y))))
    p_est_reshaped = np.reshape(p_est, (int(G.shape[0]), int(G.shape[1])))
    limit = int(np.max(np.abs(p_est_reshaped)))# / 12)
    plt.figure(figsize=(10, 20))
#plt.subplot(211)
#plt.imshow(np.real(p_est_cp_reshaped))
#plt.title("Pressure Matching at Control Points - PM")
#plt.subplot(212)
    plt.imshow(16 * np.pi * np.real(p_est_reshaped) / limit)
    plt.clim(-1, 1)
#plt.clim(-limit, limit)
    plt.colorbar()
    plt.title("Pressure Matching at Complete Sound Field - PM")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(np.real(d_hats[:, 16]))
    plt.plot(np.imag(d_hats[:, 16]))
    plt.title("Driving function PM")
    plt.show()

print("End AM")


def MM(numSPK, des, G, reg, drv0, **keyargs):
    """MM algorithm for amplitude matching
    Parameters
    ------
    numSPK: Number of loudspeakers
    des: Desired pressure
    G: Transfer function matrix
    reg: Regularization parameter
    drv0: Initial value of driving signals
    keyargs: (max_iter, dtol) = (Maximum number of iterations, Threshold for gradient of cost function)
    Returns
    ------
    drv: Loudspeaker driving signals
    drvList: List of loudspeaker driving signals for each iteration
    """
    if 'max_iter' in keyargs:
        max_iter = keyargs['max_iter']
    if 'dtol' in keyargs:
        dtol = keyargs['dtol']
    else:
        dtol = 0
    A = np.linalg.inv( G.conj().T @ G + reg * np.identity(numSPK) ) @ G.conj().T
    drv = drv0
    drvList = [drv]
    v = np.abs(des) * np.exp(1j * np.angle( G @ drv ))
    k = 0
    ddiff = 1.0
    for k in range(max_iter):
        drv = A @ v
        drvList.append(drv)
        ddiff = np.linalg.norm(drvList[k+1]-drvList[k]) / np.linalg.norm(drvList[k])
        v = np.abs(des) * np.exp(1j * np.angle( G @ drv))
        #print("itr: %d, ddiff: %f" % (k, ddiff))
        if ddiff <= dtol:
            break
    return drv, drvList

