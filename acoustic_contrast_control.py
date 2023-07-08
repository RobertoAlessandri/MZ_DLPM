import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_lib import params_linear_2D
import os
import tqdm
cp =len(params_linear_2D.idx_cp_x2_expanded * 2)

gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train' + '_nl' + str(
    params_linear_2D.N_lspks) + '_half_cp'+str(cp)+ '_xmin' + str(
    params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'
gt_soundfield_dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
    params_linear_2D.N_lspks) + '_half_cp'+str(cp)+ '_xmin' + str(
    params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'
gt_soundfield_dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train' + '_nl' + str(
    params_linear_2D.N_lspks) + '_half_cp'+str(cp)+ '_xmin' + str(
    params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'

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

check_plots = False
if check_plots:
    plt.figure(figsize=(10, 20))
    plt.subplot(311)
    plt.imshow(np.real(G[:, :, -1, 16]))
    plt.subplot(312)
    plt.imshow(np.real(G_B[:, :, -1, 16]))
    plt.subplot(313)
    plt.imshow(np.real(G_D[:, :, -1, 16]))
    plt.show()

    plt.figure(figsize=(10, 20))
    plt.subplot(311)
    plt.imshow(np.real(P_gt_[-1, :, :, 41]), cmap='coolwarm')
    plt.clim(-np.max(np.abs(P_gt_[-1, :, :, 41])), np.max(np.abs(P_gt_[-1, :, :, 41])))
    plt.subplot(312)
    plt.imshow(np.abs(P_gt__B[-1, :, :, 41]), cmap='coolwarm')
    #plt.clim(-np.max(np.abs(P_gt_[-1, :, :, 41])), np.max(np.abs(P_gt_[-1, :, :, 41])))
    plt.subplot(313)
    plt.imshow(np.angle(P_gt__B[-1, :, :, 41]), cmap='coolwarm')
    #plt.clim(-np.max(np.abs(P_gt_[-1, :, :, 41])), np.max(np.abs(P_gt_[-1, :, :, 41])))
    plt.show()

for i in range(P_gt_.shape[0]):
    for j in range(P_gt_.shape[-1]):
        P_gt_[i, int(P_gt_.shape[1] / 2):, :, j] = np.finfo(np.complex64).eps

plt.figure(figsize=(10, 20))
plt.imshow(np.real(P_gt_[-1, :, :, 16]))
plt.show()

mingrid = params_linear_2D.mingrid
if expanded:
    mingrid_B = params_linear_2D.mingrid_B_expanded
    mingrid_D = params_linear_2D.mingrid_D_expanded
else:
    mingrid_B = params_linear_2D.mingrid_B
    mingrid_D = params_linear_2D.mingrid_D

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

# ([speakers, points] * [points, speakers] + Identity).I * [speakers, points] * [points, speakers]
d_hat_list_src = []
for n_src in tqdm.tqdm(range(len(params_linear_2D.src_pos_trainT))):
    d_hats_f = []
    for n_f in range(G_cp__D.shape[2]): #tqdm.tqdm(range(G_cp__D.shape[2])):
        d_hats_eig = (np.matrix(G_cp__D[:, :, n_f]).H * np.matrix(G_cp__D[:, :, n_f]) + np.identity((np.matrix(G_cp__D[:, :, n_f]).H * np.matrix(G_cp__D[:, :, n_f])).shape[0], dtype=np.complex64)).I * np.matrix(G_cp__B[:, :, n_f]).H * np.matrix(G_cp__B[:, :, n_f])
        eigenvalues, eigenvectors = np.linalg.eig(d_hats_eig)
        index = np.argmax(eigenvalues)
        d_hat = eigenvectors[:, index]
        d_hats_f.append(d_hat)
    d_hat_list_src.append(d_hats_f)

d_hat_list_src = np.asarray(d_hat_list_src)
save=True
if save:
    np.save(os.path.join('/nas/home/ralessandri/thesis_project/dataset', 'd_hat_ACC0.npy'), d_hat_list_src)
    print("Saved")

PLOT = False
if PLOT:
    p_est_cp = d_hat_list_src[29, 16].T * G_cp_[:, :, 16].T
    p_est = d_hat_list_src[29, 16].T * G_r[:, :, 16].T
    p_est_cp_reshaped = np.reshape(p_est_cp, (int(len(c_points_x)), int(len(c_points_y))))
    p_est_reshaped = np.reshape(p_est, (int(G.shape[0]), int(G.shape[1])))
    limit = int(np.max(np.abs(p_est_reshaped))) # / 12)
    plt.figure(figsize=(10, 20))
#plt.subplot(211)
#plt.imshow(np.real(p_est_cp_reshaped))
#plt.title("Pressure Matching at Control Points - ACC")
#plt.gca().invert_yaxis()
#plt.subplot(212)
    plt.imshow(16 * np.pi * np.real(p_est_reshaped) / limit)
#plt.clim(-limit, limit)
    plt.clim(-1, 1)
    plt.colorbar()
    plt.title("Pressure Matching at Complete Sound Field - ACC")
    plt.gca().invert_yaxis()
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(np.real(d_hat))
    plt.plot(np.imag(d_hat))
    plt.title("Driving function ACC")
    plt.show()

print("End ACC")
