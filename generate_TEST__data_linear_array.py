#from numba import jit, cuda
import numpy as np
import os
import matplotlib.pyplot as plt
import sfs
import scipy
import tqdm
import argparse
from data_lib import params_linear_2D
from data_lib import soundfield_generation as sg
#import scikit-image as skimage
#from scikit-image.metrics import structural_similarity as ssim

#from skimage.metrics import structural_similarity as ssim
#plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"],
    #'font.size': 20})

idxy = params_linear_2D.idx_lr_gd_y
idx_y = idxy[int(len(idxy)/2):]

print("Running TESTing set generation")

def plot_soundfield(cmap, P, n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=True, do_norm=True):
    figure = plt.figure(figsize=(20, 20))
    if do_norm:
        print("np.shape(P) = {}, \n np.shape(P[:, n_f]) = {} ".format(np.shape(P), np.shape(P[:, n_f])))
        #im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params_linear_2D.N_sample, params_linear_2D.N_sample)), params_linear_2D.grid2D, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
        #im = sfs.plot2d.amplitude(P[:, n_f], params_linear_2D.grid2D, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    else:
        print("in else")
        #im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params_linear_2D.N_sample, params_linear_2D.N_sample)), params_linear_2D.grid2D,  cmap=cmap, colorbar=False, vmin=P[:, n_f].min(), vmax=P[:, n_f].max(), xnorm=None)
        #im = sfs.plot2d.amplitude(P[:, n_f], params_linear_2D.grid2D, xnorm=None, cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    if plot_ldspks:
        sfs.plot2d.loudspeakers(params_linear_2D.array.x[selection], params_linear_2D.array.n[selection], a0=1, size=0.18)
    plt.xlabel('$x [m]$', fontsize=axis_label_size), plt.ylabel('$y [m]$', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    #cbar = plt.colorbar(im, fraction=0.046)
    #cbar.ax.tick_params(labelsize=tick_font_size)
    # cbar.set_label('$NRE~[\mathrm{dB}]$',fontsize=tick_font_size))
    plt.tight_layout()
    if save_path is not None:
        #plt.savefig(save_path)
        print("save_path")
    plt.show()

#@jit(target_backend='cuda')
def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='/nas/home/ralessandri/thesis_project/dataset')
    #parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/dataset')
    ##parser.add_argument('--dataset_name', type=str, help="Base Data Directory", default='data_src_wideband_point_W_23_test')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers', default=0)
    args = parser.parse_args()
    propagate_filters = False
    eval_points = False
    control_points = True
    PLOT_RESULTS = False

    if control_points:
        c_points_x = params_linear_2D.idx_cp_x2
        c_points_y = params_linear_2D.idx_cp_y2

        c_pointsx_y = c_points_x[int(len(c_points_x) / 2):]

    dataset_path = '/nas/home/ralessandri/thesis_project/dataset/test'
    #dataset_path = 'C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/dataset/linear_array'

    # Setup
    # Grid of points where we actually compute the soundfield
    point = params_linear_2D.point
    #N_pts = len(point)
    N_pts = int((len(params_linear_2D.grid2D[0][0])))
    grid = params_linear_2D.grid2D

    # Secondary Sources Green function
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_linear_2D.N_lspks) + '_r_' + str(
        params_linear_2D.rangeX[0]) + '.npy'
    if os.path.exists(os.path.join(dataset_path, green_function_sec_sources_path)):
        G = np.load(os.path.join(dataset_path, green_function_sec_sources_path))
    else:
        G = np.zeros((N_pts, N_pts, params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
        for n_l in tqdm.tqdm(range(params_linear_2D.N_lspks)):
            for n_f in range(len(params_linear_2D.f_axis)):
                # hankel_factor_1 = params_linear_2D.wc[n_f] / params_linear_2D.c  # , (params_linear_2D.N_lspks, 1)
                # print("array_pos = {}\nhankel_factor_2 = point[np] - array_pos = {}".format(params_linear_2D.array_pos, point - params_linear_2D.array_pos))
                # hankel_factor_2 = np.linalg.norm(grid - params_linear_2D.array_pos[n_l])  # , reps=(params_linear_2D.N_freqs, 1)  # ).transpose()
                k = params_linear_2D.wc[n_f] / params_linear_2D.c
                r = np.linalg.norm(grid - params_linear_2D.array_pos[n_l])
                # np.exp(-1j * k * r) / (4 * np.pi)
                # Points, Speakers, Frequencies
                # G[:, :, n_l, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
                into_G = np.exp(-1j * k * r) / (4 * np.pi)
                # print("into_G = {}".format(np.shape(into_G)))
                G[:, :, n_l, n_f] = into_G  # / r # ! Sto volutamente non tenendo conto del decay!

        print("G is saved, shape = --> {}".format(np.shape(G)))
        np.save(os.path.join(dataset_path, green_function_sec_sources_path), G)

    # Check if array in grid points are equal
    for n_p in range(N_pts):
        if np.sum(np.linalg.norm(point[n_p] - params_linear_2D.array_pos, axis=1) == 0) > 0:
            print(str(n_p))

    N_missing = args.n_missing
    N_lspks = params_linear_2D.N_lspks - N_missing
    if N_missing>0:
        lspks_config_path = 'lspk_config_nl_'+str(params_linear_2D.N_lspks)+'_missing_'+str(N_missing)+'.npy'
        lspk_config_path_global = os.path.join(dataset_path, 'setup', lspks_config_path)
        if os.path.exist(lspk_config_path_global):
            idx_missing = np.load(lspk_config_path_global)
            print('Loaded existing mic config')
        else:
            idx_missing = np.random.choice(np.arange(params_linear_2D.N_lspks), size=N_missing, replace=False)
            print("saved missing indexes, shape --> {}".format(np.shape(idx_missing)))
            np.save(lspk_config_path_global, idx_missing)
        theta_l = np.delete(params_linear_2D.theta_l, idx_missing)
        G = np.delete(G, idx_missing, axis=1)

    if eval_points:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(params_linear_2D.idx_lr_gd_y)), int(len(params_linear_2D.idx_lr_gd_x)), params_linear_2D.N_freqs), dtype=complex)
    elif control_points:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(c_points_y)), int(len(c_points_x)), params_linear_2D.N_freqs), dtype=complex)
    else:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), N_pts, N_pts, params_linear_2D.N_freqs),dtype=complex)  # 3204, 64


    d_array = np.zeros((params_linear_2D.N_lspks, N_lspks, params_linear_2D.N_freqs), dtype=complex)
    for n_s in tqdm.tqdm(range(len(params_linear_2D.src_pos_testT))):

        xs = np.append(params_linear_2D.src_pos_testT[n_s], 2)

        # Multiply filters for hergoltz density function
        if propagate_filters:
            P = np.zeros((N_pts, len(params_linear_2D.wc)), dtype=complex)
            for n_p in range(N_pts):
                P[n_p, :] = np.sum(G[n_p] * d_array[n_s], axis=0)

        if args.gt_soundfield:
            for n_f in range(params_linear_2D.N_freqs):
                # Sources, Points, Frequencies
                #P_gt[n_s, :, :, n_f] = (1j / 4) * scipy.special.hankel2(0, (params_linear_2D.wc[n_f] / params_linear_2D.c) * np.linalg.norm(point[:, :2] - xs))
                if eval_points:
                    into_P_gt_eval = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, grid) * np.linalg.norm(grid - xs)
                    for i in params_linear_2D.idx_lr_gd_x:
                        for j in idx_y:
                            into_P_gt_eval[-j, i] = 0
                    into_P_gt_eval_y = into_P_gt_eval[params_linear_2D.idx_lr_gd_y]
                    P_gt[n_s, :, :, n_f] = into_P_gt_eval_y[:,params_linear_2D.idx_lr_gd_x]

                elif control_points:
                    into_P_gt_eval = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, grid) * np.linalg.norm(grid - xs)  # this last multiplication is not valid if I consider the decay!
                    for i in c_points_x:
                        for j in c_pointsx_y:
                            into_P_gt_eval[-j, i] = 0
                    into_P_gt_eval_y = into_P_gt_eval[c_points_y]
                    P_gt[n_s, :, :, n_f] = into_P_gt_eval_y[:, c_points_x]

                else:
                    P_gt[n_s, :, :, n_f] = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, grid) * np.linalg.norm(grid - xs)


    if PLOT_RESULTS:
            # Plot params
            print("Plotting")
            selection = np.ones_like(params_linear_2D.array_pos[:, 0])
            selection = selection == 1  # ?
            #n_s = 32
            n_f = 41
            print(str(params_linear_2D.f_axis[n_f]))
            cmap = 'RdBu_r'
            tick_font_size = 70
            axis_label_size = 90
            plot_soundfield(cmap, P_gt[n_s], n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=True,
                            do_norm=False)

    if args.gt_soundfield:
        if control_points:
            print("P_gt saved shape --> {}".format(np.shape(P_gt)))
            np.save(os.path.join(dataset_path, 'gt_soundfield_train_half_cp_double_train.npy'), P_gt)
        else:
            print("P_gt saved shape --> {}".format(np.shape(P_gt)))
            np.save(os.path.join(dataset_path, 'gt_soundfield_train.npy'), P_gt)

    #np.save(os.path.join(dataset_path, 'filters_config_nl_'+str(N_lspks)+'_missing_'+str(N_missing)+'.npy'))

    # P_gt = [sources, control points, frequencies]
    #print("x = {}\ny = {}".format(np.shape(P_gt[:, :, 0]), np.shape(P_gt[0, :, :])))
    #plt.figure(figsize=(10,10))
    #plt.plot(P_gt[:64, :, 0].T, P_gt[0, :, :], 'g*')
    #plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    #plt.xlim(-2, 2)
    #plt.ylim(-2, 2)
    ##plt.setzlim(0, 4)
    ##plt.legend(['eval points', 'control points', 'loudspeakers', 'train sources', 'test sources'])
    #plt.title("#2D")
    #plt.show()

print("Everything Defined")
if __name__ == '__main__':
    print("Main running")
    main()
    print("##   End Generate TESTing   ##")


