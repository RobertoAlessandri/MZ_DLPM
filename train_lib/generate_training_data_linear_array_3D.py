import numpy as np
import os
import matplotlib.pyplot as plt
import sfs
import scipy
import tqdm
import argparse
from data_lib import params_linear_3D
from data_lib import soundfield_generation as sg
#import scikit-image as skimage
#from scikit-image.metrics import structural_similarity as ssim

#from skimage.metrics import structural_similarity as ssim
#plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"],
    #'font.size': 20})

def plot_soundfield(cmap, P, n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=True, do_norm=True):
    figure = plt.figure(figsize=(20, 20))
    if do_norm:
        print("np.shape(P) = {}, \n np.shape(P[:, n_f]) = {} ".format(np.shape(P), np.shape(P[:, n_f])))
        im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params_linear_3D.N_sample, params_linear_3D.N_sample)), params_linear_3D.grid3D, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
        #im = sfs.plot2d.amplitude(P[:, n_f], params_linear_3D.grid2D, xnorm=[0, 0, 0], cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    else:
        print("in else")
        #im = sfs.plot2d.amplitude(np.reshape(P[:, n_f], (params_linear_3D.N_sample, params_linear_3D.N_sample)), params_linear_3D.grid3D,  cmap=cmap, colorbar=False, vmin=P[:, n_f].min(), vmax=P[:, n_f].max(), xnorm=None)
        #im = sfs.plot2d.amplitude(P[:, n_f], params_linear_3D.grid2D, xnorm=None, cmap=cmap, vmin=-1.0, vmax=1.0, colorbar=False)
    if plot_ldspks:
        sfs.plot2d.loudspeakers(params_linear_3D.array.x[selection], params_linear_3D.array.n[selection], a0=1, size=0.18)
    plt.xlabel('$x [m]$', fontsize=axis_label_size), plt.ylabel('$y [m]$', fontsize=axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_font_size)
    cbar = plt.colorbar(im, fraction=0.046)
    cbar.ax.tick_params(labelsize=tick_font_size)
    # cbar.set_label('$NRE~[\mathrm{dB}]$',fontsize=tick_font_size))
    plt.tight_layout()
    if save_path is not None:
        #plt.savefig(save_path)
        print("save_path")
    plt.show()


def main():
    # Arguments parse
    parser = argparse.ArgumentParser(description='Generate data for linear array setup')

    parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='/nas/home/ralessandri/thesis_project/dataset')
    #parser.add_argument('--base_dir', type=str, help="Base Data Directory", default='C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/dataset')

    ##parser.add_argument('--dataset_name', type=str, help="Base Data Directory", default='data_src_wideband_point_W_23_train')
    ##parser.add_argument('--dataset_path', type=str, help='path to dataset', default='/nas/home/lcomanducci/soundfield_synthesis/dataset/data_src_wideband_point_W_23_train.npz' )
    parser.add_argument('--gt_soundfield', type=bool, help='compute ground truth soundfield',
                        default=True)
    parser.add_argument('--n_missing', type=int, help='number missing loudspeakers',
                        default=0)
    args = parser.parse_args()
    eval_points = True
    PLOT_RESULTS = False

    dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array'
    #dataset_path = 'C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/dataset/linear_array'

    # Setup
    # Grid of points where we actually compute the soundfield
    point = params_linear_3D.point

    N_pts = len(point)
    print("N_pts = ", N_pts)

    # Secondary Sources Green function
    green_function_sec_sources_path = 'green_function_sec_sources_nl_' + str(params_linear_3D.N_lspks) + '_r_' + str(params_linear_3D.rangeX[0]) + '_3D_' +'.npy'
    if os.path.exists(os.path.join(dataset_path, green_function_sec_sources_path)):
        G = np.load(os.path.join(dataset_path, green_function_sec_sources_path))
    else:
        #2097152, 16, 64
        G = np.zeros((N_pts, params_linear_3D.N_lspks, params_linear_3D.N_freqs), dtype=complex)
        for n_p in tqdm.tqdm(range(N_pts)):
            hankel_factor_1 = np.tile(params_linear_3D.wc / params_linear_3D.c, (params_linear_3D.N_lspks, 1))
            hankel_factor_2 = np.tile(np.linalg.norm(point[n_p] - params_linear_3D.array_pos, axis=1), reps=(params_linear_3D.N_freqs, 1)).transpose()
            G[n_p, :, :] = (1j / 4) * scipy.special.hankel2(0, hankel_factor_1*hankel_factor_2)
        np.save(os.path.join(dataset_path, green_function_sec_sources_path), G)

    for n_p in range(N_pts):
        if np.sum(np.linalg.norm(point[n_p] - params_linear_3D.array_pos, axis=1) == 0) > 0:
            print(str(n_p))


    if eval_points:
        N_pts = len(params_linear_3D.idx_cp)
        print("\nG shape = {}".format(np.shape(G)))
        G = G[params_linear_3D.idx_lr[params_linear_3D.idx_cp]]
        point = params_linear_3D.point_cp


    print("P_gt should be\n[{}, {}, {}]".format(len(params_linear_3D.src_pos_train), N_pts, params_linear_3D.N_freqs))
    P_gt = np.zeros((len(params_linear_3D.src_pos_train), N_pts, params_linear_3D.N_freqs), dtype=complex)
    # 16, 1600, 64

    print("Entering in the for cycle")
    print("len(params_linear_3D.sources_positions) = {} ".format(len(params_linear_3D.src_pos_train)))
    print("range(len(params_linear_3D.sources_positions)) = {} ".format(range(len(params_linear_3D.src_pos_train))))
    print("tqdm.tqdm(range(len(params_linear_3D.sources_positions))) = {} ".format(tqdm.tqdm(range(len(params_linear_3D.src_pos_train)))))
    for n_s in tqdm.tqdm(range(len(params_linear_3D.src_pos_train))):
        #print("Entered in the for cycle")
        xs = params_linear_3D.src_pos_train[n_s]

        # np.tile(np.expand_dims(Phi,axis=0),reps=(params_circular.N_lspks,1,1))
        # Ground truth source
        if args.gt_soundfield:
            for n_f in range(params_linear_3D.N_freqs):
                P_gt[n_s, :, n_f] = (1j / 4) * \
                               scipy.special.hankel2(0,
                                                     (params_linear_3D.wc[n_f] / params_linear_3D.c) *
                                                     np.linalg.norm(point[:, :2] - xs, axis=1))
        #print("will it plot?")
        if PLOT_RESULTS:
            # Plot params
            print("Plotting")
            selection = np.ones_like(params_linear_3D.array_pos[:, 0])
            selection = selection == 1  # ?
            #n_s = 32
            n_f = 41
            print(str(params_linear_3D.f_axis[n_f]))
            cmap = 'RdBu_r'
            tick_font_size = 70
            axis_label_size = 90
            plot_soundfield(cmap, P_gt[n_s], n_f, selection, axis_label_size, tick_font_size, save_path=None, plot_ldspks=True,
                            do_norm=False)
            print('bella')

    if args.gt_soundfield:
        np.save(os.path.join(dataset_path, 'gt_soundfield_train.npy'), P_gt)

print("Everything Defined")
if __name__ == '__main__':
    print("Main running")
    main()
    print("##   End Generate Training 3D ##")

