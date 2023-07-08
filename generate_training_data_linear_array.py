import numpy as np
import os
import matplotlib.pyplot as plt
import sfs
import tqdm
import argparse
from data_lib import params_linear_2D

#from skimage.metrics import structural_similarity as ssim
#plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"],
    #'font.size': 20})

print("Running training set generation")
local = False
#@jit(target_backend='cuda')
def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')
    if local:
        parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/dataset')
    else:
        parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='/nas/home/ralessandri/thesis_project/dataset')
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield', default=True)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers', default=0)
    args = parser.parse_args()
    eval_points = False
    control_points = True
    switch = False
    expanded = True

    if switch:
        print("\nSWITCHED\n")
    if expanded:
        print("\nEXPANDED\n")

    # indexes for evaluation points
    if expanded:
        idxy = params_linear_2D.idx_lr_gd_y_expanded
        idx_y = idxy[int(len(idxy) / 2):]  # ?
    else:
        idxy = params_linear_2D.idx_lr_gd_y
        idx_y = idxy[int(len(idxy) / 2):]  # ?

    if control_points:
        if expanded:
            c_points_x = params_linear_2D.idx_cp_x2_expanded
            c_points_y = params_linear_2D.idx_cp_y2_expanded
            c_pointsx_y = c_points_y[int(len(c_points_y) / 2):]  # ?
        else:
            c_points_x = params_linear_2D.idx_cp_x2
            c_points_y = params_linear_2D.idx_cp_y2
            c_pointsx_y = c_points_y[int(len(c_points_y) / 2):]  # ?

    if local:
        dataset_path = 'C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/dataset/linear_array'
    else:
        dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array'
        dataset_path_bright = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright'
        dataset_path_dark = '/nas/home/ralessandri/thesis_project/dataset/linear_array/dark'



    ## Setup ##
    # Grid of points where we actually compute the soundfield
    point = params_linear_2D.point
    grid = params_linear_2D.grid2D
    N_pts = int((len(grid[0][0])))
    N_lspks = params_linear_2D.N_lspks
    PLOT_RESULTS = False


    mingrid = params_linear_2D.mingrid
    if switch:
        mingrid_B = params_linear_2D.mingrid_B_switched
        mingrid_D = params_linear_2D.mingrid_D_switched
    elif expanded:
        mingrid_B = params_linear_2D.mingrid_B_expanded
        mingrid_D = params_linear_2D.mingrid_D_expanded
    else:
        mingrid_B = params_linear_2D.mingrid_B
        mingrid_D = params_linear_2D.mingrid_D


    # Secondary Sources Green function
    if switch:
        green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(N_lspks) + '_r_' + str(params_linear_2D.rangeX[0]) + '_decay_switch.npy'
    elif expanded:
        green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(N_lspks) + '_r_' + str(params_linear_2D.rangeX_expanded[0]) + '_decay_expanded_fs15.npy'
    else:
        green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(N_lspks) + '_r_' + str(params_linear_2D.rangeX[0]) + '_decay.npy'
    if os.path.exists(os.path.join(dataset_path, green_function_sec_sources_path)):
        G = np.load(os.path.join(dataset_path, green_function_sec_sources_path))
    else:
        G = np.zeros((N_pts, N_pts, N_lspks, params_linear_2D.N_freqs), dtype=complex)
        for n_l in tqdm.tqdm(range(N_lspks)):
            for n_f in range(len(params_linear_2D.f_axis)):
                #hankel factors were useful if we were working in 3D
                #hankel_factor_1 = params_linear_2D.wc[n_f] / params_linear_2D.c  # , (params_linear_2D.N_lspks, 1)
                #hankel_factor_2 = np.linalg.norm(grid - params_linear_2D.array_pos[n_l])  # , reps=(params_linear_2D.N_freqs, 1)  # ).transpose()
                k = params_linear_2D.wc[n_f] / params_linear_2D.c  # wave number
                r = np.linalg.norm(grid - params_linear_2D.array_pos[n_l])  # distance speaker-to-point
                # Points, Speakers, Frequencies
                #G[:, :, n_l, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
                into_G = np.exp(-1j * k * r) / (4 * np.pi * r)
                G[:, :, n_l, n_f] = into_G

        np.save(os.path.join(dataset_path, green_function_sec_sources_path), G)
        print("G is saved, shape --> {}".format(np.shape(G)))

    if os.path.exists(os.path.join(dataset_path_bright, green_function_sec_sources_path)):
        G_B = np.load(os.path.join(dataset_path_bright, green_function_sec_sources_path))
    else:
        G_B = np.zeros((int(len(mingrid_B[0][0])), int(len(mingrid_B[1])), N_lspks, params_linear_2D.N_freqs), dtype=complex)
        for n_l in tqdm.tqdm(range(N_lspks)):
            for n_f in range(len(params_linear_2D.f_axis)):
                # hankel factors were useful if we were working in 3D
                # hankel_factor_1 = params_linear_2D.wc[n_f] / params_linear_2D.c  # , (params_linear_2D.N_lspks, 1)
                # hankel_factor_2 = np.linalg.norm(grid - params_linear_2D.array_pos[n_l])  # , reps=(params_linear_2D.N_freqs, 1)  # ).transpose()
                k = params_linear_2D.wc[n_f] / params_linear_2D.c  # wave number
                r_B = np.linalg.norm(mingrid_B - params_linear_2D.array_pos[n_l])  # distance speaker-to-point
                # Points, Speakers, Frequencies
                # G[:, :, n_l, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
                into_G_B = np.exp(-1j * k * r_B) / (4 * np.pi * r_B)
                G_B[:, :, n_l, n_f] = into_G_B

        np.save(os.path.join(dataset_path_bright, green_function_sec_sources_path), G_B)
        print("G_B is saved, shape --> {}".format(np.shape(G_B)))

    if os.path.exists(os.path.join(dataset_path_dark, green_function_sec_sources_path)):
        G_D = np.load(os.path.join(dataset_path_dark, green_function_sec_sources_path))
    else:
        G_D = np.zeros((int(len(mingrid_D[0][0])), int(len(mingrid_D[1])), N_lspks, params_linear_2D.N_freqs), dtype=complex)
        for n_l in tqdm.tqdm(range(N_lspks)):
            for n_f in range(len(params_linear_2D.f_axis)):
                # hankel factors were useful if we were working in 3D
                # hankel_factor_1 = params_linear_2D.wc[n_f] / params_linear_2D.c  # , (params_linear_2D.N_lspks, 1)
                # hankel_factor_2 = np.linalg.norm(grid - params_linear_2D.array_pos[n_l])  # , reps=(params_linear_2D.N_freqs, 1)  # ).transpose()
                k = params_linear_2D.wc[n_f] / params_linear_2D.c  # wave number
                r_D = np.linalg.norm(mingrid_D - params_linear_2D.array_pos[n_l])  # distance speaker-to-point
                # Points, Speakers, Frequencies
                # G[:, :, n_l, n_f] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
                into_G_D = np.exp(-1j * k * r_D) / (4 * np.pi * r_D)
                G_D[:, :, n_l, n_f] = into_G_D

        np.save(os.path.join(dataset_path_dark, green_function_sec_sources_path), G_D)
        print("G_D is saved, shape --> {}".format(np.shape(G_D)))


    # checking green functions are correct, comment to accelerate generation
    if PLOT_RESULTS:
        plt.figure(figsize=(10, 20))
        plt.subplot(211)
        plt.imshow((into_G / r).astype(np.float))
        plt.show()
        plt.subplot(212)
        sfs.plot2d.amplitude(2 * (into_G / r) / np.max(np.abs(into_G / r)) , mingrid, colorbar_kwargs=dict(label="p / Pa"))
        plt.title("Speaker {} and frequency {}".format(n_l,params_linear_2D.f_axis[n_f]))
        plt.show()



    # Check if array in grid points are equal
    for n_p in range(N_pts):
        if np.sum(np.linalg.norm(point[n_p] - params_linear_2D.array_pos) == 0) > 0:
            print("str(n_p) = ", str(n_p))

    # defining ground truths
    if eval_points:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(params_linear_2D.idx_lr_gd_y)), int(len(params_linear_2D.idx_lr_gd_x)), params_linear_2D.N_freqs), dtype=complex)
    elif control_points:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(c_points_y)), int(len(c_points_x)), params_linear_2D.N_freqs), dtype=complex)
        P_gt_B = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(mingrid_B[0][0])), int(len(mingrid_B[1])), params_linear_2D.N_freqs), dtype=complex)
        P_gt_D = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(mingrid_D[0][0])), int(len(mingrid_D[1])), params_linear_2D.N_freqs), dtype=complex)
    else:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), N_pts, N_pts, params_linear_2D.N_freqs),dtype=complex)  # 3204, 64

    for n_s in tqdm.tqdm(range(len(params_linear_2D.src_pos_trainT))):

        xs = np.append(params_linear_2D.src_pos_trainT[n_s], 2)  # position in 3D

        # Ground truth source
        if args.gt_soundfield:
            for n_f in range(params_linear_2D.N_freqs):

                # Sources, Points, Frequencies
                if eval_points:
                    into_P_gt_eval = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, grid)
                    for i in params_linear_2D.idx_lr_gd_x:
                        for j in idx_y:
                            into_P_gt_eval[-j, i] = 0 #np.finfo(np.complex128).eps
                    into_P_gt_eval_y = into_P_gt_eval[params_linear_2D.idx_lr_gd_y]
                    P_gt[n_s, :, :, n_f] = into_P_gt_eval_y[:, params_linear_2D.idx_lr_gd_x]

                elif control_points:
                    # This commented part was used when we weren't separating completely bright and dark zone
                    into_P_gt_cp = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, grid, c=params_linear_2D.c)
                    for i in c_points_x:
                        for j in c_pointsx_y:
                            into_P_gt_cp[i, j] = np.finfo(np.complex64).eps
                    into_P_gt_cp_y = into_P_gt_cp[c_points_y]
                    P_gt[n_s, :, :, n_f] = into_P_gt_cp_y[:, c_points_x]

                    into_P_gt_cp_B = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, mingrid_B, c=params_linear_2D.c)
                    into_P_gt_cp_D = np.zeros(np.shape(into_P_gt_cp_B)) + np.finfo(np.complex64).eps
                    P_gt_B[n_s, :, :, n_f] = into_P_gt_cp_B
                    P_gt_D[n_s, :, :, n_f] = into_P_gt_cp_D

                    # checking the soundfield is correct
                    if PLOT_RESULTS:
                        normalization_point = 4 * np.pi
                        selection = np.ones_like(params_linear_2D.array_pos[:, 0])
                        plt.figure(figsize=(10, 20))
                        plt.subplot(211)
                        show = into_P_gt_cp_y[:, c_points_x]
                        plt.imshow(show.astype(np.float))
                        plt.subplot(212) # * normalization_point
                        sfs.plot2d.amplitude(2 * show / np.max(np.abs(show)) , mingrid, colorbar_kwargs=dict(label="p / Pa"))
                        sfs.plot2d.loudspeakers(params_linear_2D.array.x, params_linear_2D.array.n, selection)
                        plt.title("Point Source at {} m and frequency {}".format(xs,params_linear_2D.f_axis[n_f]))
                        plt.show()

                else:
                    P_gt[n_s, :, :, n_f] = sfs.fd.source.point(params_linear_2D.wc[n_f], xs, grid) * np.linalg.norm(grid - xs)

    if args.gt_soundfield:
        if control_points:
            if switch:
                np.save(os.path.join(dataset_path, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_switch.npy'), P_gt)
                print("P_gt saved, shape --> {}".format(np.shape(P_gt)))
                np.save(os.path.join(dataset_path_bright, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_switch.npy'), P_gt_B)
                print("P_gt_B saved, shape --> {}".format(np.shape(P_gt_B)))
                np.save(os.path.join(dataset_path_dark, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_switch.npy'), P_gt_D)
                print("P_gt_D saved, shape --> {}".format(np.shape(P_gt_D)))
            elif expanded:
                np.save(os.path.join(dataset_path, 'gt_soundfield_TESTiNV'+'_nl'+str(N_lspks)+'_half_cp'+str(len(params_linear_2D.idx_cp_x2_expanded*2))+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'), P_gt)
                print("P_gt saved, shape --> {}".format(np.shape(P_gt)))
                np.save(os.path.join(dataset_path_bright, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp'+str(len(params_linear_2D.idx_cp_x2_expanded*2))+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'), P_gt_B)
                print("P_gt_B saved, shape --> {}".format(np.shape(P_gt_B)))
                np.save(os.path.join(dataset_path_dark, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp'+str(len(params_linear_2D.idx_cp_x2_expanded*2))+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'), P_gt_D)
                print("P_gt_D saved, shape --> {}".format(np.shape(P_gt_D)))
            else:
                np.save(os.path.join(dataset_path, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy'), P_gt)
                print("P_gt saved, shape --> {}".format(np.shape(P_gt)))
                np.save(os.path.join(dataset_path_bright, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy'), P_gt_B)
                print("P_gt_B saved, shape --> {}".format(np.shape(P_gt_B)))
                np.save(os.path.join(dataset_path_dark, 'gt_soundfield_train'+'_nl'+str(N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy'), P_gt_D)
                print("P_gt_D saved, shape --> {}".format(np.shape(P_gt_D)))
        elif eval_points:
            np.save(os.path.join(dataset_path, 'gt_soundfield_train_eval_points.npy'), P_gt)
            print("P_gt saved shape --> {}".format(np.shape(P_gt)))
        else:
            np.save(os.path.join(dataset_path, 'gt_soundfield_train.npy'), P_gt)
            print("P_gt saved shape --> {}".format(np.shape(P_gt)))


if __name__ == '__main__':
    print("Main running")
    main()
    print("##   End Generate Training   ##")


