import numpy as np
import matplotlib.pyplot as plt
import sfs
import scipy
import tqdm
import argparse
import tensorflow as tf
from data_lib import params_linear_2D
#import jax.numpy as jnp
import os
import train_sep_B_D
#import pressure_matching
#import acoustic_contrast_control
#import Amplitude_Matching_MinMax
# FIX PRESSURE MATCHING SCRIPT
# FIX IDENTATION NMSE



from skimage.metrics import structural_similarity as ssim

def normalize(x):
    min_x = x.min()
    max_x = x.max()
    den = max_x-min_x
    if den==0:
        den += np.finfo(np.float16).eps
    x_norm = (x - min_x)/(den)
    return x_norm

def nmse(P_hat, P_gt,type='freq'):
    if type=='freq':
        return np.mean((np.power(np.abs(P_hat[ :, :] - P_gt[ :, :]), 2)), axis=0)
    else:
        return np.power(np.abs(P_hat[ :, :] - P_gt[ :, :]), 2)

def nmseAbs(P_hat, P_gt,type='freq'):
    if type=='freq':
        return np.mean((np.power(np.abs(np.abs(P_hat[ :, :]) - np.abs(P_gt[ :, :])), 2)), axis=0)
    else:
        return np.power(np.abs(P_hat[ :, :] - P_gt[ :, :]), 2)

def nmseAngle(P_hat, P_gt,type='freq'):
    if type=='freq':
        return np.mean((np.power(np.abs(np.angle(P_hat[ :, :]) - np.angle(P_gt[ :, :])), 2)), axis=0)
    else:
        return np.power(np.abs(P_hat[ :, :] - P_gt[ :, :]), 2)

def AC(P_B, P_D):
    """Average Acoustic Contrast (AC)
    Parameters
    ------
    synB: Synthesized pressure in bright zone
    synQ: Synthesized pressure in quiet zone
    Returns
    ------
    AC: Average Acoustic Contrast (AC)
    """

    #AC = 10 * np.log10( np.sum( np.linalg.norm(P_B, axis=1)**2 ) / np.sum( np.linalg.norm(P_D, axis=1)**2 ) )
    #AC = 10 * np.log10(np.sum(P_B[:, :] ** 2, axis=0) / np.sum(P_D[:, :] ** 2, axis=0))
    AC = 10 * np.log10(np.real(np.sum(np.conj(P_B)*P_B, axis=0) / np.sum(np.conj(P_D)*P_D, axis=0)))

    return AC

def ssim_abs(P_hat, P_gt):
    P_hat = normalize(np.abs(P_hat))
    P_gt = normalize(np.abs(P_gt))
    return ssim(P_gt, P_hat, data_range=1)

def ssim_freq(P_hat, P_gt):
    ssim_freq_array = np.zeros(params_linear_2D.N_freqs)
    for n_f in range(params_linear_2D.N_freqs):
        ssim_freq_array[n_f] = ssim_abs(P_hat[:, n_f], P_gt[:, n_f])
    return ssim_freq_array

def main():
    # Arguments parse
    print("start evaluation metric")

    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    parser.add_argument('--dataset_path', type=str, help="Base Data Directory",
                            default='/nas/home/ralessandri/thesis_project/dataset/linear_array')
    parser.add_argument('--model_path', type=str, help='Deep learning models folder',
                            default='/nas/home/ralessandri/thesis_project/models/linear_array')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    parser.add_argument('--pm', type=bool, help='compute pressure matching', default=True)
    parser.add_argument('--pwd_cnn', type=bool, help='compute model-based acoustic rendering + CNN', default=True)
    parser.add_argument('--n_loudspeakers', type=int, help='Numbr of loudspeakers in array', default=params_linear_2D.N_lspks)
    eval_points = True
    PLOT = False
    PLOT_DRIVING = False
    args = parser.parse_args()
    dataset_path = args.dataset_path


    # Load green function secondary sources --> eval points (it is in train directory since it is the same)
    dataset_path_train = '/nas/home/ralessandri/thesis_project/dataset/linear_array'
    # FAI OS JOIN
    green_function_path = 'green_function_sec_sources_nl_' + str(args.n_loudspeakers) + '_r_' + str(params_linear_2D.rangeX_expanded[0]) + '_decay_expanded_fs15.npy'
                        # green_function_sec_sources_nl_64_r_-0.3_decay_expanded_fs15.npy
    green_function_path_bright = 'bright/green_function_sec_sources_nl_' + str(args.n_loudspeakers) + '_r_' + str(params_linear_2D.rangeX_expanded[0]) + '_decay_expanded_fs15.npy'
    green_function_path_dark = 'dark/green_function_sec_sources_nl_' + str(args.n_loudspeakers) + '_r_' + str(params_linear_2D.rangeX_expanded[0]) + '_decay_expanded_fs15.npy'


    G = np.load(os.path.join(dataset_path, green_function_path))  # green function
    G_B = np.load(os.path.join(dataset_path, green_function_path_bright))  # green function bright zone
    G_D = np.load(os.path.join(dataset_path, green_function_path_dark))  # green function dark zone
    G_Bravel = np.zeros((G_B.shape[0] * G_B.shape[1], G_B.shape[2], G_B.shape[3]), dtype=np.complex64)
    G_Dravel = np.zeros((G_D.shape[0] * G_D.shape[1], G_D.shape[2], G_D.shape[3]), dtype=np.complex64)
    G_ravel = np.zeros((40401, G_D.shape[2], G_D.shape[3]), dtype=np.complex64)
    for n_f in range(64):
        for n_r in range(64):
            G_ravel[:, n_r, n_f] = np.ravel(G[:, :, n_r, n_f])
            G_Bravel[:, n_r, n_f] = np.ravel(G_B[:, :, n_r, n_f])
            G_Dravel[:, n_r, n_f] = np.ravel(G_D[:, :, n_r, n_f])


    # Let's precompute what we need in order to apply the selected models
    # Load pwd_cnn deep learning model
    expanded=True
    if expanded:
        c_points_x = params_linear_2D.idx_cp_x2_expanded
        c_points_y = params_linear_2D.idx_cp_y2_expanded
    else:
        c_points_x = params_linear_2D.idx_cp_x2
        c_points_y = params_linear_2D.idx_cp_y2

    lr = 0.003
    lambda_abs = 12.5 + np.finfo(dtype=np.float16).eps
    lambda_D = 12.5 + np.finfo(dtype=np.float16).eps#/ lambda_abs + np.finfo(dtype=np.float16).eps

    if args.pwd_cnn:
        model_name = 'model_linear_config_nl_' + str(args.n_loudspeakers)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay_expanded_fs15_TEST_INVreIm'
        network_model = tf.keras.models.load_model(os.path.join(args.model_path, model_name))




    if args.pm:
        #d_hats_pm = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_PM.npy')
        d_hats_pm = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_PM_Switch.npy')

        d_hats_acc = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_ACC.npy')
        #d_hats_am = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_AM.npy')
        d_hats_am = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_AM150.npy')


        d_hats_acc0 = np.load('/nas/home/ralessandri/thesis_project/dataset/d_hat_ACC0.npy')



    if eval_points:  # probabilmente avrò errori
        N_pts = len(params_linear_2D.idx_lr)
        G_cp_y = G[params_linear_2D.idx_lr_gd_y]
        G_lr_ = G_cp_y[:, params_linear_2D.idx_lr_gd_x]



    #nmse_pwd_cnn = np.zeros((len(params_linear_2D.src_pos_trainT), params_linear_2D.virtual_s_x, params_linear_2D.virtual_s_y, params_linear_2D.N_freqs))
    nmse_pwd_cnn = np.zeros((150, params_linear_2D.N_freqs))
    nmse_pwd_cnn_B = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_cnn_D = np.zeros_like(nmse_pwd_cnn)

    nmse_pwd_pm = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_pm_B = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_pm_D = np.zeros_like(nmse_pwd_cnn)

    nmse_pwd_acc = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_acc_B = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_acc_D = np.zeros_like(nmse_pwd_cnn)

    nmse_pwd_am = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_am_B = np.zeros_like(nmse_pwd_cnn)
    nmse_pwd_am_D = np.zeros_like(nmse_pwd_cnn)


    nmseAbs_pwd_cnn = np.zeros((150, params_linear_2D.N_freqs))
    nmseAbs_pwd_cnn_B = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_cnn_D = np.zeros_like(nmse_pwd_cnn)

    nmseAbs_pwd_pm = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_pm_B = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_pm_D = np.zeros_like(nmse_pwd_cnn)

    nmseAbs_pwd_acc = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_acc_B = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_acc_D = np.zeros_like(nmse_pwd_cnn)

    nmseAbs_pwd_am = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_am_B = np.zeros_like(nmse_pwd_cnn)
    nmseAbs_pwd_am_D = np.zeros_like(nmse_pwd_cnn)


    nmseAngle_pwd_cnn = np.zeros((150, params_linear_2D.N_freqs))
    nmseAngle_pwd_cnn_B = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_cnn_D = np.zeros_like(nmse_pwd_cnn)

    nmseAngle_pwd_pm = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_pm_B = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_pm_D = np.zeros_like(nmse_pwd_cnn)

    nmseAngle_pwd_acc = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_acc_B = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_acc_D = np.zeros_like(nmse_pwd_cnn)

    nmseAngle_pwd_am = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_am_B = np.zeros_like(nmse_pwd_cnn)
    nmseAngle_pwd_am_D = np.zeros_like(nmse_pwd_cnn)


    ssim_pwd_cnn = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_cnn_B = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_cnn_D = np.zeros_like(nmse_pwd_cnn)

    ssim_pwd_pm = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_pm_B = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_pm_D = np.zeros_like(nmse_pwd_cnn)

    ssim_pwd_acc = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_acc_B = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_acc_D = np.zeros_like(nmse_pwd_cnn)

    ssim_pwd_am = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_am_B = np.zeros_like(nmse_pwd_cnn)
    ssim_pwd_am_D = np.zeros_like(nmse_pwd_cnn)


    ac_pwd_cnn = np.zeros_like(nmse_pwd_cnn)

    ac_pwd_pm = np.zeros_like(nmse_pwd_cnn)

    ac_pwd_acc = np.zeros_like(nmse_pwd_cnn)

    ac_pwd_am = np.zeros_like(nmse_pwd_cnn)



    mingrid_B = params_linear_2D.mingrid_B
    mingrid_D = params_linear_2D.mingrid_D


    P_gt = np.zeros(
        (len(params_linear_2D.src_pos_trainT), 450,
            params_linear_2D.N_freqs), dtype=complex)
    # 961 instead int(len(mingrid_B[0][0]) / 2) * int(len(mingrid_B[1]) / 2)
    P_gt_B = np.zeros(
        (len(params_linear_2D.src_pos_trainT), 961,
            params_linear_2D.N_freqs), dtype=complex)
    P_gt_input = np.zeros(
        (len(params_linear_2D.src_pos_trainT), 225,
            params_linear_2D.N_freqs), dtype=complex)
    P_gt_D = np.zeros(
        (len(params_linear_2D.src_pos_trainT), 961,
            params_linear_2D.N_freqs), dtype=complex)

    cp = len(params_linear_2D.idx_cp_x2_expanded * 2)
    if args.gt_soundfield:
        gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp' + str(cp) + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'
        gt_soundfield_dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp' + str(cp) + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'
        gt_soundfield_dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train' + '_nl' + str(
            params_linear_2D.N_lspks) + '_half_cp' + str(cp) + '_xmin' + str(
            params_linear_2D.x_min_train) + '_decay_expanded_fs15.npy'

        P_gt_ = np.load(gt_soundfield_dataset_path)  # gt soundfield
        P_gt__B = np.load(gt_soundfield_dataset_path_bright)  # gt soundfield bright zone
        P_gt__D = np.load(gt_soundfield_dataset_path_dark)  # gt soundfield dark zon

        #P_gt_y = P_gt_[params_linear_2D.idx_lr_gd_y]
        #P_gt_lr = P_gt_y[:, params_linear_2D.idx_lr_gd_x]

    for i in range(len(params_linear_2D.src_pos_trainT)):
        for j in range(params_linear_2D.N_freqs):
            P_to_ravel = P_gt_[i, :, :, j]
            P_to_ravel_B = P_gt__B[i, :, :, j]
            P_to_ravel_input = P_gt__B[i, 0:-1:2, 0:-1:2, j]
            P_to_ravel_D = P_gt__D[i, :, :, j]

            P_gt[i, :, j] = np.ravel(P_to_ravel)
            P_gt_B[i, :, j] = np.ravel(P_to_ravel_B)
            P_gt_input[i, :, j] = np.ravel(P_to_ravel_input)
            P_gt_D[i, :, j] = np.ravel(P_to_ravel_D)

    P_gt150 = P_gt[0:-1:10, :, :]
    P_gt_B150 = P_gt_B[0:-1:10, :, :]
    P_gt_input150 = P_gt_input[0:-1:10, :, :]
    P_gt_D150 = P_gt_D[0:-1:10, :, :]



    N_pts = P_gt_B.shape[1]

    for n_r in tqdm.tqdm(range(len(params_linear_2D.src_pos_trainT[0:-1:10]))):
        #n_r_AM = n_r % 10

        P_pwd_cnn = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_pm = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_acc = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_am = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)

        P_pwd_cnn_B = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_pm_B = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_acc_B = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_am_B = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)


        P_pwd_cnn_D = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_pm_D = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_acc_D = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
        P_pwd_am_D = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)


        P_pwd_acc0 = np.zeros((40401, params_linear_2D.N_freqs), dtype=complex)


        #xs = params_linear_2D.src_pos_train[n_r].append(2)
        PLOT_TEST=False
        if PLOT_TEST:
            n_r = -1
        if args.pwd_cnn:
            if eval_points:
                P_input = P_gt_input[n_r]
            else:
                P_input = P_gt_input[params_linear_2D.idx_lr[params_linear_2D.idx_cp]]



            d_array_cnn = network_model.predict(
                np.expand_dims(np.concatenate([np.real(P_input), np.imag(P_input)], axis=0),
                                axis=[0, -1]).astype('float32'))[0, :, :, 0].astype('float64')

            d_array_cnn_complex = d_array_cnn[:int(d_array_cnn.shape[0] / 2)] + (
                        1j * d_array_cnn[int(d_array_cnn.shape[0] / 2):])

            if PLOT_DRIVING:
                plt.figure(figsize=(10, 60))
                plt.subplot(611)
                plt.imshow(np.real(d_array_cnn[:64, :].T), aspect='auto', cmap='RdBu')
                plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.title("Driving function_NN_real")
                plt.subplot(612)
                plt.imshow(np.real(d_hats_pm[:, :, 0].T), aspect='auto', cmap='RdBu')
                plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.title("Driving function_PM_real")
                plt.subplot(613)
                plt.imshow(np.real(d_hats_acc[:, :, 0].T), aspect='auto', cmap='RdBu')
                plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.title("Driving function_ACC_real")
                plt.subplot(614)
                plt.imshow(np.real(d_array_cnn[64:, :].T), aspect='auto', cmap='RdBu')
                plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.title("Driving function_NN_imag")
                plt.subplot(615)
                plt.imshow(np.imag(d_hats_pm[:, :, 0].T), aspect='auto', cmap='RdBu')
                plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.title("Driving function_PM_imag")
                plt.subplot(616)
                plt.imshow(np.imag(d_hats_acc[:, :, 0].T), aspect='auto', cmap='RdBu')
                plt.xlabel('$l$', fontsize=120), plt.ylabel('$k$', fontsize=120)
                plt.tick_params(axis='both', which='major', labelsize=120)
                plt.gca().invert_yaxis()
                plt.title("Driving function_PM_imag")

            plt.show()
            #n_f=41



            checkNow = True
            if checkNow:
                N_pts = 40401
                P_pwd_cnn = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
                P_pwd_pm = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
                P_pwd_acc = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)
                P_pwd_am = np.zeros((N_pts, params_linear_2D.N_freqs), dtype=complex)

                for n__p in range(40401):
                    #P_pwd_cnn[n__p, :] = np.sum(G_ravel[n__p] * d_array_cnn_complex, axis=0)
                    P_pwd_pm[n__p, :] = np.sum(G_ravel[n__p] * d_hats_pm[n_r, :, :, 0].T, axis=0)
                    #P_pwd_acc[n__p, :] = np.sum(G_ravel[n__p] * d_hats_acc[n_r, :, :, 0].T, axis=0)
                    P_pwd_acc[n__p, :] = np.sum(G_ravel[n__p] * d_hats_acc0[n_r, :, :, 0].T, axis=0)
                    if n_r % 10 == 0:
                        P_pwd_am[n__p, :] = np.sum(G_ravel[n__p] * d_hats_am[n_r, :, :, 0].T, axis=0)

                n_f = 41
                P_pwd_cnn[:, :] = tf.einsum('ij,kij->kj', d_array_cnn_complex,
                                               tf.cast(G_ravel, dtype=tf.complex64))

                #plt.figure(figsize=(10, 10))
                #plt.imshow(np.real(P_gt_[n_r, :, :, n_f] / np.abs(P_gt_[n_r, 15, 7, n_f])), cmap='coolwarm')
                #plt.title('GT')
                #plt.clim(-2, 2)
                #plt.show()

                ax = plt.figure(figsize=(10, 40))
                plt.subplot(411)
                plt.imshow(np.real(np.reshape(P_pwd_cnn[:, n_f], (201, 201))), cmap='coolwarm')
                plt.title('CNN')
                plt.clim(-0.2, 0.2)
                plt.colorbar()
                plt.subplot(412)
                plt.imshow(np.real(np.reshape(P_pwd_pm[:, n_f], (201, 201)) ), cmap='coolwarm')
                plt.title('PM')
                plt.clim(-0.2, 0.2)
                plt.colorbar()
                plt.subplot(413)
                plt.imshow(np.real(np.reshape(P_pwd_acc[:, n_f], (201, 201))), cmap='coolwarm')
                plt.title('ACC')
                plt.clim(-0.2, 0.2)
                plt.colorbar()
                #plt.imshow(np.real(np.reshape(P_pwd_acc0[:, n_f], (201, 201))), cmap='coolwarm')
                #plt.title('ACC0')
                #plt.clim(-0.2, 0.2)
                #plt.show()
                plt.subplot(414)
                plt.imshow(np.real(np.reshape(P_pwd_am[:, n_f], (201, 201))) , cmap='coolwarm')
                plt.clim(-0.2, 0.2)
                plt.colorbar()
                plt.show()

                plt.figure(figsize=(10, 40))
                plt.subplot(411)
                plt.imshow(np.real(np.reshape(P_pwd_cnn[:, n_f], (201, 201))/ np.abs(P_pwd_cnn[20201, n_f])), cmap='coolwarm')
                plt.title('CNN')
                plt.clim(-2, 2)
                plt.colorbar()
                plt.subplot(412)
                plt.imshow(np.real(np.reshape(P_pwd_pm[:, n_f], (201, 201)) / np.abs(P_pwd_pm[20201, n_f])), cmap='coolwarm')
                plt.title('PM')
                plt.clim(-2, 2)
                plt.colorbar()
                plt.subplot(413)
                plt.imshow(np.real(np.reshape(P_pwd_acc[:, n_f], (201, 201)) / np.abs(P_pwd_acc[20201, n_f])), cmap='coolwarm')
                plt.title('ACC')
                plt.clim(-2, 2)
                plt.colorbar()
                #plt.imshow(np.real(np.reshape(P_pwd_acc0[:, n_f], (201, 201))), cmap='coolwarm')
                #plt.title('ACC0')
                #plt.clim(-0.2, 0.2)
                #plt.show()
                plt.subplot(414)
                plt.imshow(np.real(np.reshape(P_pwd_am[:, n_f], (201, 201)) / np.abs(P_pwd_am[20201, n_f])), cmap='coolwarm')
                plt.title('AM')
                plt.clim(-2, 2)
                plt.colorbar()
                plt.show()

                plt.figure(figsize=(10, 40))
                plt.subplot(411)
                plt.imshow(np.abs(np.reshape(P_pwd_cnn[:, n_f], (201, 201))/ np.abs(P_pwd_cnn[20201, n_f])), cmap='coolwarm')
                plt.title('Abs - CNN')
                plt.clim(0, 4)
                plt.colorbar()
                plt.subplot(412)
                plt.imshow(np.abs(np.reshape(P_pwd_pm[:, n_f], (201, 201)) / np.abs(P_pwd_pm[20201, n_f])), cmap='coolwarm')
                plt.title('Abs - PM')
                plt.clim(0, 4)
                plt.colorbar()
                plt.subplot(413)
                plt.imshow(np.abs(np.reshape(P_pwd_acc[:, n_f], (201, 201)) / np.abs(P_pwd_acc[20201, n_f])), cmap='coolwarm')
                plt.title('Abs - ACC')
                plt.clim(0, 4)
                plt.colorbar()
                #plt.imshow(np.real(np.reshape(P_pwd_acc0[:, n_f], (201, 201))), cmap='coolwarm')
                #plt.title('ACC0')
                #plt.clim(-0.2, 0.2)
                #plt.show()
                plt.subplot(414)
                plt.imshow(np.abs(np.reshape(P_pwd_am[:, n_f], (201, 201)) / np.abs(P_pwd_am[20201, n_f])), cmap='coolwarm')
                plt.title('Abs - AM')
                plt.clim(0, 4)
                plt.colorbar()
                plt.show()

                plt.figure(figsize=(10, 40))
                plt.subplot(411)
                plt.imshow(20*np.log10(np.abs(np.reshape(P_pwd_cnn[:, n_f], (201, 201)) / np.abs(P_pwd_cnn[20201, n_f]))),
                           cmap='coolwarm_clip', vmax=3, vmin=-50)
                plt.title('Abs [dB] - CNN')
                plt.clim(-60, 60)
                plt.colorbar()
                plt.subplot(412)
                plt.imshow(20*np.log10(np.abs(np.reshape(P_pwd_pm[:, n_f], (201, 201)) / np.abs(P_pwd_pm[20201, n_f]))),
                           cmap='coolwarm_clip', vmax=3, vmin=-50)
                plt.title('Abs [dB] - PM')
                plt.clim(-60, 60)
                plt.colorbar()
                plt.subplot(413)
                plt.imshow(20*np.log10(np.abs(np.reshape(P_pwd_acc[:, n_f], (201, 201)) / np.abs(P_pwd_acc[20201, n_f]))),
                           cmap='coolwarm_clip', vmax=3, vmin=-50)
                plt.title('Abs [dB] - ACC')
                plt.clim(-60, 60)
                plt.colorbar()
                # plt.imshow(np.real(np.reshape(P_pwd_acc0[:, n_f], (201, 201))), cmap='coolwarm')
                # plt.title('ACC0')
                # plt.clim(-0.2, 0.2)
                # plt.show()
                plt.subplot(414)
                plt.imshow(20*np.log10(np.abs(np.reshape(P_pwd_am[:, n_f], (201, 201)) / np.abs(P_pwd_am[20201, n_f]))),
                           cmap='coolwarm_clip', vmax=3, vmin=-50)
                plt.title('Abs [dB] - AM')
                plt.clim(-60, 60)
                plt.colorbar()
                plt.show()

                plt.figure(figsize=(10, 40))
                plt.subplot(411)
                plt.imshow(np.angle(np.reshape(P_pwd_cnn[:, n_f], (201, 201))/ np.angle(P_pwd_cnn[20201, n_f])), cmap='coolwarm')
                plt.title('Angle - CNN')
                #plt.clim(-2, 2)
                plt.subplot(412)
                plt.imshow(np.angle(np.reshape(P_pwd_pm[:, n_f], (201, 201)) / np.angle(P_pwd_pm[20201, n_f])), cmap='coolwarm')
                plt.title('Angle - PM')
                #plt.clim(-2, 2)
                plt.subplot(413)
                plt.imshow(np.angle(np.reshape(P_pwd_acc[:, n_f], (201, 201)) / np.angle(P_pwd_acc[20201, n_f])), cmap='coolwarm')
                plt.title('Angle - ACC')
                #plt.clim(-2, 2)
                #plt.imshow(np.real(np.reshape(P_pwd_acc0[:, n_f], (201, 201))), cmap='coolwarm')
                #plt.title('ACC0')
                #plt.clim(-0.2, 0.2)
                #plt.show()
                plt.subplot(414)
                plt.imshow(np.angle(np.reshape(P_pwd_am[:, n_f], (201, 201)) / np.angle(P_pwd_am[20201, n_f])), cmap='coolwarm')
                plt.title('Angle - AM')
                #plt.clim(-2, 2)
                plt.show()

            for n_p in range(N_pts):
                P_pwd_cnn[n_p, :] = np.sum(G_ravel[n_p] * d_array_cnn_complex, axis=0)
                P_pwd_cnn_B[n_p, :] = np.sum(G_Bravel[n_p] * d_array_cnn_complex, axis=0)
                P_pwd_cnn_D[n_p, :] = np.sum(G_Dravel[n_p] * d_array_cnn_complex, axis=0)

                #P_pwd_pm[n_p, :] = np.sum(G_ravel[n_p] * d_hats_pm[n_r, :, :,0], axis=0)
                P_pwd_pm_B[n_p, :] = np.sum(G_Bravel[n_p] * d_hats_pm[n_r, :, :,0], axis=0)
                P_pwd_pm_D[n_p, :] = np.sum(G_Dravel[n_p] * d_hats_pm[n_r, :, :,0], axis=0)

                #P_pwd_acc[n_p, :] = np.sum(G_ravel[n_p] * d_hats_acc[n_r, :, :,0], axis=0)
                P_pwd_acc_B[n_p, :] = np.sum(G_Bravel[n_p] * d_hats_acc[n_r, :, :,0].T, axis=0)
                P_pwd_acc_D[n_p, :] = np.sum(G_Dravel[n_p] * d_hats_acc[n_r, :, :,0].T, axis=0)

                #P_pwd_am[n_p, :] = np.sum(G_ravel[n_p] * d_hats_am[n_r, :, :,0], axis=0)
                P_pwd_am_B[n_p, :] = np.sum(G_Bravel[n_p] * d_hats_am[n_r, :, :,0], axis=0)
                P_pwd_am_D[n_p, :] = np.sum(G_Dravel[n_p] * d_hats_am[n_r, :, :,0], axis=0)

                #P_pwd_cnn[n_p, :] = np.sum(G_lr_[n_p] * d_array_cnn_complex, axis=0)

                ################# Remember that I only need the evaluation points !!! ###############################

            #for n_p in range(N_pts):


                #P_pwd_pm[n_p, :] = np.sum(G[n_p] * d_hats[:, :,0], axis=0)

        nmse_pwd_cnn[n_r], nmse_pwd_pm[n_r], nmse_pwd_acc[n_r], nmse_pwd_am[n_r] = nmse(np.concatenate((P_pwd_cnn_B, P_pwd_cnn_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmse(np.concatenate((P_pwd_pm_B, P_pwd_pm_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmse(np.concatenate((P_pwd_acc_B, P_pwd_acc_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmse(np.concatenate((P_pwd_am_B, P_pwd_am_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0))
        nmseAbs_pwd_cnn[n_r], nmseAbs_pwd_pm[n_r], nmseAbs_pwd_acc[n_r], nmseAbs_pwd_am[n_r] = nmseAbs(np.concatenate((P_pwd_cnn_B, P_pwd_cnn_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmseAbs(np.concatenate((P_pwd_pm_B, P_pwd_pm_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmseAbs(np.concatenate((P_pwd_acc_B, P_pwd_acc_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmseAbs(np.concatenate((P_pwd_am_B, P_pwd_am_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0))
        nmseAngle_pwd_cnn[n_r], nmseAngle_pwd_pm[n_r], nmseAngle_pwd_acc[n_r], nmseAngle_pwd_am[n_r] = nmseAngle(np.concatenate((P_pwd_cnn_B, P_pwd_cnn_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmseAngle(np.concatenate((P_pwd_pm_B, P_pwd_pm_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmseAngle(np.concatenate((P_pwd_acc_B, P_pwd_acc_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), nmseAngle(np.concatenate((P_pwd_am_B, P_pwd_am_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0))
        ssim_pwd_cnn[n_r], ssim_pwd_pm[n_r], ssim_pwd_acc[n_r], ssim_pwd_am[n_r] = ssim_freq(np.concatenate((P_pwd_cnn_B, P_pwd_cnn_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), ssim_freq(np.concatenate((P_pwd_pm_B, P_pwd_pm_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), ssim_freq(np.concatenate((P_pwd_acc_B, P_pwd_acc_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0)), ssim_freq(np.concatenate((P_pwd_am_B, P_pwd_am_D), axis=0), np.concatenate((P_gt_B[n_r], P_gt_D[n_r]), axis=0))
        ac_pwd_cnn[n_r], ac_pwd_pm[n_r], ac_pwd_acc[n_r], ac_pwd_am[n_r] = AC(P_pwd_cnn_B, P_pwd_cnn_D), AC(P_pwd_pm_B, P_pwd_pm_D), AC(P_pwd_acc_B, P_pwd_acc_D), AC(P_pwd_am_B, P_pwd_am_D)


        nmse_pwd_cnn_B[n_r], nmse_pwd_pm_B[n_r], nmse_pwd_acc_B[n_r], nmse_pwd_am_B[n_r],  = nmse(P_pwd_cnn_B, P_gt_B[n_r]), nmse(P_pwd_pm_B, P_gt_B[n_r]), nmse(P_pwd_acc_B, P_gt_B[n_r]), nmse(P_pwd_am_B, P_gt_B[n_r])
        nmseAbs_pwd_cnn_B[n_r], nmseAbs_pwd_pm_B[n_r], nmseAbs_pwd_acc_B[n_r], nmseAbs_pwd_am_B[n_r] = nmseAbs(P_pwd_cnn_B, P_gt_B[n_r]), nmseAbs(P_pwd_pm_B, P_gt_B[n_r]), nmseAbs(P_pwd_acc_B, P_gt_B[n_r]), nmseAbs(P_pwd_am_B, P_gt_B[n_r])
        nmseAngle_pwd_cnn_B[n_r], nmseAngle_pwd_pm_B[n_r], nmseAngle_pwd_acc_B[n_r], nmseAngle_pwd_am_B[n_r] = nmseAngle(P_pwd_cnn_B, P_gt_B[n_r]), nmseAngle(P_pwd_pm_B, P_gt_B[n_r]), nmseAngle(P_pwd_acc_B, P_gt_B[n_r]), nmseAngle(P_pwd_am_B, P_gt_B[n_r])
        ssim_pwd_cnn_B[n_r], ssim_pwd_pm_B[n_r], ssim_pwd_acc_B[n_r], ssim_pwd_am_B[n_r] = ssim_freq(P_pwd_cnn_B, P_gt_B[n_r]), ssim_freq(P_pwd_pm_B, P_gt_B[n_r]), ssim_freq(P_pwd_acc_B, P_gt_B[n_r]), ssim_freq(P_pwd_am_B, P_gt_B[n_r])

        nmse_pwd_cnn_D[n_r], nmse_pwd_pm_D[n_r], nmse_pwd_acc_D[n_r], nmse_pwd_am_D[n_r] = nmse(P_pwd_cnn_D, P_gt_D[n_r]), nmse(P_pwd_pm_D, P_gt_D[n_r]), nmse(P_pwd_acc_D, P_gt_D[n_r]), nmse(P_pwd_am_D, P_gt_D[n_r])
        nmseAbs_pwd_cnn_D[n_r], nmseAbs_pwd_pm_D[n_r], nmseAbs_pwd_acc_D[n_r], nmseAbs_pwd_am_D[n_r] = nmseAbs(P_pwd_cnn_D, P_gt_D[n_r]), nmseAbs(P_pwd_pm_D, P_gt_D[n_r]), nmseAbs(P_pwd_acc_D, P_gt_D[n_r]), nmseAbs(P_pwd_am_D, P_gt_D[n_r])
        nmseAngle_pwd_cnn_D[n_r], nmseAngle_pwd_pm_D[n_r], nmseAngle_pwd_acc_D[n_r], nmseAngle_pwd_am_D[n_r] = nmseAngle(P_pwd_cnn_D, P_gt_D[n_r]), nmseAngle(P_pwd_pm_D, P_gt_D[n_r]), nmseAngle(P_pwd_acc_D, P_gt_D[n_r]), nmseAngle(P_pwd_am_D, P_gt_D[n_r])
        ssim_pwd_cnn_D[n_r], ssim_pwd_pm_D[n_r], ssim_pwd_acc_D[n_r], ssim_pwd_am_D[n_r] = ssim_freq(P_pwd_cnn_D, P_gt_D[n_r]), ssim_freq(P_pwd_pm_D, P_gt_D[n_r]), ssim_freq(P_pwd_acc_D, P_gt_D[n_r]), ssim_freq(P_pwd_am_D, P_gt_D[n_r])

        if PLOT:
            #plt.figure(),\
            #plt.plot(params_linear_2D.array_pos[:4, 0],
            #         params_linear_2D.array_pos[:4, 1],
            #                        'r*'),\
            #plt.show()

            d = np.linalg.norm(
                np.array([params_linear_2D.array_pos[1, 0], params_linear_2D.array_pos[1, 1]]) - np.array(
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
            save_path_gt = os.path.join(plot_paths, 'sf_real_source_'
                                        + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(
                args.n_loudspeakers) + '.pdf')

            # PWD-CNN
            save_path_pwd_cnn = os.path.join(plot_paths, 'sf_pwd_cnn_'
                                        + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(
                args.n_loudspeakers) + '.pdf')

            # Error
            nmse_pwd_cnn = 10 * np.log10(nmse(P_pwd_cnn_B, P_gt_B[n_r], type='full'))
            save_path_error_cnn = os.path.join(plot_paths, 'nmse_pwd_cnn_'
                                        + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(
                args.n_loudspeakers) + '.pdf')

            # PM
            save_path_pm = os.path.join(plot_paths, 'sf_pm_'
                                        + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(
                args.n_loudspeakers) + '.pdf')

            # Error
            nmse_pm = 10 * np.log10(nmse(P_pwd_pm_B, P_gt_B[n_r], type='full'))
            save_path_error_pm = os.path.join(plot_paths, 'nmse_pm_'
                                        + str(n_r) + '_f_' + str(params_linear_2D.f_axis[n_f]) + '_nl' + str(
                args.n_loudspeakers) + '.pdf')

            plt.figure(figsize=(20,40))
            plt.subplot(411)
            plt.plot(params_linear_2D.f_axis, 10 * np.log10(np.mean(nmse(P_pwd_cnn_B, P_gt_B[n_r], type='full'), axis=0)),
                        'k-*')
            plt.plot(params_linear_2D.f_axis, 10 * np.log10(np.mean(nmse(P_pwd_pm_B, P_gt_B[n_r], type='full'), axis=0)),
                        'r-*')
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.ylabel('NMSE dB')
            plt.xlabel('Frequency [Hz]')
            plt.legend(['CNN', 'PM'])
            plt.title("NMSE_Bright")

            plt.subplot(412)
            plt.plot(params_linear_2D.f_axis, ssim_freq(P_pwd_cnn_B, P_gt_B[n_r]),
                        'k-*')
            plt.plot(params_linear_2D.f_axis, ssim_freq(P_pwd_pm_B, P_gt_B[n_r]),
                        'r-*')
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.ylabel('SSIM')
            plt.xlabel('Frequency [Hz]')
            plt.legend(['CNN', 'PM'])
            plt.title("SSIM_Bright")

            plt.subplot(413)
            plt.plot(params_linear_2D.f_axis, 10 * np.log10(np.mean(nmse(P_pwd_cnn_D, P_gt_D[n_r], type='full'), axis=0)),
                        'k-*')
            plt.plot(params_linear_2D.f_axis, 10 * np.log10(np.mean(nmse(P_pwd_pm_D, P_gt_D[n_r], type='full'), axis=0)),
                        'r-*')
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.ylabel('NMSE dB')
            plt.xlabel('Frequency [Hz]')
            plt.legend(['CNN', 'PM'])
            plt.title("NMSE_Dark")

            plt.subplot(414)
            plt.plot(params_linear_2D.f_axis, ssim_freq(P_pwd_cnn_D, P_gt_D[n_r]),
                        'k-*')
            plt.plot(params_linear_2D.f_axis, ssim_freq(P_pwd_pm_D, P_gt_D[n_r]),
                        'r-*')
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.ylabel('SSIM')
            plt.xlabel('Frequency [Hz]')
            plt.legend(['CNN', 'PM'])
            plt.title("SSIM_Dark")

            plt.show()

            print('pause')

    # If we are plotting it means we are just computing data for the paper --> no need to save anything
    if not PLOT:
        # Save arrays
        np.savez(os.path.join(dataset_path, 'nmse_nl_' + str(args.n_loudspeakers) + '.npz'),
                 nmse_pwd_cnn=nmse_pwd_cnn, nmse_pwd_pm=nmse_pwd_pm, nmse_pwd_acc=nmse_pwd_acc, nmse_pwd_am=nmse_pwd_am)
        np.savez(os.path.join(dataset_path, 'nmseAbs_nl_' + str(args.n_loudspeakers) + '.npz'),
                 nmseAbs_pwd_cnn=nmseAbs_pwd_cnn, nmseAbs_pwd_pm=nmseAbs_pwd_pm, nmseAbs_pwd_acc=nmseAbs_pwd_acc, nmseAbs_pwd_am=nmseAbs_pwd_am)
        np.savez(os.path.join(dataset_path, 'nmseAngle_nl_' + str(args.n_loudspeakers) + '.npz'),
                 nmseAngle_pwd_cnn=nmseAngle_pwd_cnn, nmseAngle_pwd_pm=nmseAngle_pwd_pm, nmseAngle_pwd_acc=nmseAngle_pwd_acc, nmseAngle_pwd_am=nmseAngle_pwd_am)
        np.savez(os.path.join(dataset_path, 'ssim_nl_' + str(args.n_loudspeakers) + '.npz'),
                    ssim_pwd_cnn=ssim_pwd_cnn, ssim_pwd_pm=ssim_pwd_pm, ssim_pwd_acc=ssim_pwd_acc, ssim_pwd_am=ssim_pwd_am)
        np.savez(os.path.join(dataset_path, 'ac_nl_' + str(args.n_loudspeakers) + '.npz'),
                    ac_pwd_cnn=ac_pwd_cnn, ac_pwd_pm=ac_pwd_pm, ac_pwd_acc=ac_pwd_acc, ac_pwd_am=ac_pwd_am)

        np.savez(os.path.join(dataset_path, 'nmse_B_nl_' + str(args.n_loudspeakers) + '.npz'),
                    nmse_pwd_cnn=nmse_pwd_cnn_B, nmse_pwd_pm=nmse_pwd_pm_B, nmse_pwd_acc=nmse_pwd_acc_B, nmse_pwd_am=nmse_pwd_am_B)
        np.savez(os.path.join(dataset_path, 'nmseAbs_B_nl_' + str(args.n_loudspeakers) + '.npz'),
                    nmseAbs_pwd_cnn=nmseAbs_pwd_cnn_B, nmseAbs_pwd_pm=nmseAbs_pwd_pm_B, nmseAbs_pwd_acc=nmseAbs_pwd_acc_B, nmseAbs_pwd_am=nmseAbs_pwd_am_B)
        np.savez(os.path.join(dataset_path, 'nmseAngle_B_nl_' + str(args.n_loudspeakers) + '.npz'),
                    nmseAngle_pwd_cnn=nmseAngle_pwd_cnn_B, nmseAngle_pwd_pm=nmseAngle_pwd_pm_B, nmseAngle_pwd_acc=nmseAngle_pwd_acc_B, nmseAngle_pwd_am=nmseAngle_pwd_am_B)
        np.savez(os.path.join(dataset_path, 'ssim_B_nl_' + str(args.n_loudspeakers) + '.npz'),
                    ssim_pwd_cnn=ssim_pwd_cnn_B, ssim_pwd_pm=ssim_pwd_pm_B, ssim_pwd_acc=ssim_pwd_acc_B, ssim_pwd_am=ssim_pwd_am_B)

        np.savez(os.path.join(dataset_path, 'nmse_D_nl_' + str(args.n_loudspeakers) + '.npz'),
                    nmse_pwd_cnn=nmse_pwd_cnn_D, nmse_pwd_pm=nmse_pwd_pm_D, nmse_pwd_acc=nmse_pwd_acc_D, nmse_pwd_am=nmse_pwd_am_D)
        np.savez(os.path.join(dataset_path, 'nmseAbs_D_nl_' + str(args.n_loudspeakers) + '.npz'),
                    nmseAbs_pwd_cnn=nmseAbs_pwd_cnn_D, nmseAbs_pwd_pm=nmseAbs_pwd_pm_D, nmseAbs_pwd_acc=nmseAbs_pwd_acc_D, nmseAbs_pwd_am=nmseAbs_pwd_am_D)
        np.savez(os.path.join(dataset_path, 'nmseAngle_D_nl_' + str(args.n_loudspeakers) + '.npz'),
                    nmseAngle_pwd_cnn=nmseAngle_pwd_cnn_D, nmseAngle_pwd_pm=nmseAngle_pwd_pm_D, nmseAngle_pwd_acc=nmseAngle_pwd_acc_D, nmseAngle_pwd_am=nmseAngle_pwd_am_D)
        np.savez(os.path.join(dataset_path, 'ssim_D_nl_' + str(args.n_loudspeakers) + '.npz'),
                    ssim_pwd_cnn=ssim_pwd_cnn_D, ssim_pwd_pm=ssim_pwd_pm_D, ssim_pwd_acc=ssim_pwd_acc_D, ssim_pwd_am=ssim_pwd_am_D)

if __name__ == '__main__':
    main()
    print("end evaluation metric")
