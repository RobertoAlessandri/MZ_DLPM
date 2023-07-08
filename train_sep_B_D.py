import numpy as np
import os
import argparse
import datetime
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['CUDA_ALLOW_GROWTH']='True'
import tensorflow as tf
#from tensorboard.plugins.hparams import api as hp # SCOMMENTA !!! <--
import matplotlib.pyplot as plt
import sfs
from train_lib import network_utils
from train_lib import train_utils
from data_lib import params_linear_2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

print("tensorflow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) != 0:
    print("tf.test.gpu_device_name() = ",tf.test.gpu_device_name())
    print("tf.config.list_physical_devices('GPU') = ", tf.config.list_physical_devices('GPU'))

AUTOTUNE = tf.data.experimental.AUTOTUNE

print("Running train")

def main():
    control_points = True
    eval_points = False
    test_semplice = False
    local = False
    switch = False
    expanded = True

    # parsers
    parser = argparse.ArgumentParser(description='Sounfield reconstruction')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    if local:
        parser.add_argument('--log_dir', type=str, help='Tensorboard log directory',
                            default='C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/logs/scalars')
        parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset',
                            default='C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/dataset/linear_array/gt_soundfield_train'+'_nl'+str(
            params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy')
    else:
        parser.add_argument('--log_dir', type=str, help='Tensorboard log directory', default='/nas/home/ralessandri/thesis_project/logs/scalars')
    parser.add_argument('--number_missing_ldspk', type=int, help='number of missing loudspeakers',default=0)
    if control_points:
        if switch:
            print("\nSWITCHED model\n")
            parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_switch.npy')
            parser.add_argument('--gt_soundfield_dataset_path_bright', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_TESTiNV'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_switch.npy')
            parser.add_argument('--gt_soundfield_dataset_path_dark', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_switch.npy')
        elif expanded:
            parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp'+str(len(params_linear_2D.idx_cp_x2_expanded*2))+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy')
            parser.add_argument('--gt_soundfield_dataset_path_bright', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp'+str(len(params_linear_2D.idx_cp_x2_expanded*2))+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy')
            parser.add_argument('--gt_soundfield_dataset_path_dark', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp'+str(len(params_linear_2D.idx_cp_x2_expanded*2))+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy')
        else:
            parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy')
            parser.add_argument('--gt_soundfield_dataset_path_bright', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy')
            parser.add_argument('--gt_soundfield_dataset_path_dark', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/gt_soundfield_train'+'_nl'+str(
                params_linear_2D.N_lspks)+'_half_cp_double_train_0_y+1'+'_xmin'+str(params_linear_2D.x_min_train)+'_decay.npy')
    elif eval_points:
        parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset', default='/nas/home/ralessandri/thesis_project/dataset/linear_array/gt_soundfield_train.npy' )
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=3*0.001) # put 2! * 0.001
    parser.add_argument('--n_loudspeakers', type=float, help='Number Loudspeakers', default=params_linear_2D.N_lspks)
    if local:
        parser.add_argument('--green_function', type=str, help='Green Function', default='C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/dataset/linear_array/green_function_sec_sources' + '_nl_' + str(
                                params_linear_2D.N_lspks) + '_r_-0.25_decay.npy')
    else:
        if expanded:
            parser.add_argument('--green_function', type=str, help='Green Function',
                                default='/nas/home/ralessandri/thesis_project/dataset/linear_array/green_function_sec_sources' + '_nl_' + str(
                                    params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy')
            parser.add_argument('--green_function_bright', type=str, help='Green Function',
                                default='/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/green_function_sec_sources' + '_nl_' + str(
                                    params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy')
            parser.add_argument('--green_function_dark', type=str, help='Green Function',
                                default='/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/green_function_sec_sources' + '_nl_' + str(
                                    params_linear_2D.N_lspks) + '_r_-0.3_decay_expanded_fs15.npy')
        else:
            parser.add_argument('--green_function', type=str, help='Green Function',
                                default='/nas/home/ralessandri/thesis_project/dataset/linear_array/green_function_sec_sources' + '_nl_' + str(
                                    params_linear_2D.N_lspks) + '_r_-0.25_decay.npy')
            parser.add_argument('--green_function_bright', type=str, help='Green Function',
                                default='/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/green_function_sec_sources' + '_nl_' + str(
                                    params_linear_2D.N_lspks) + '_r_-0.25_decay.npy')
            parser.add_argument('--green_function_dark', type=str, help='Green Function',
                                default='/nas/home/ralessandri/thesis_project/dataset/linear_array/dark/green_function_sec_sources' + '_nl_' + str(
                                    params_linear_2D.N_lspks) + '_r_-0.25_decay.npy')
    parser.add_argument('--gpu', type=str, help='gpu number', default='1')

    # parameters definition
    args = parser.parse_args()
    number_missing_loudspeakers = args.number_missing_ldspk
    epochs = args.epochs
    batch_size = args.batch_size
    log_dir = args.log_dir
    N_lspks = args.n_loudspeakers

    gt_soundfield_dataset_path = args.gt_soundfield_dataset_path
    gt_soundfield_dataset_path_bright = args.gt_soundfield_dataset_path_bright
    gt_soundfield_dataset_path_dark = args.gt_soundfield_dataset_path_dark


    lr = args.learning_rate
    green_function_path = args.green_function
    green_function_path_bright = args.green_function_bright
    green_function_path_dark = args.green_function_dark

    # Ausiliary grids
    if expanded:
        mingrid_B = params_linear_2D.mingrid_B_expanded
        mingrid_D = params_linear_2D.mingrid_D_expanded

        c_points_x = params_linear_2D.idx_cp_x2_expanded
        c_points_y = params_linear_2D.idx_cp_y2_expanded

    else:
        if switch:
            mingrid_B = params_linear_2D.mingrid_B_switched
            mingrid_D = params_linear_2D.mingrid_D_switched
        else:
            mingrid_B = params_linear_2D.mingrid_B
            mingrid_D = params_linear_2D.mingrid_D

        c_points_x = params_linear_2D.idx_cp_x2
        c_points_y = params_linear_2D.idx_cp_y2
        mingrid = params_linear_2D.mingrid


    c_points_x_ = np.arange(len(mingrid_B[0][0]))[0:-1:2]
    c_points_y_ = np.arange(len(mingrid_D[1]))[0:-1:2]

    # Empirically assessed
    # 3 and 2 are best values
    lambda_abs = 12.5 + np.finfo(dtype=np.float16).eps # we weight the absolute value loss since it is in a different range w.r.t. phase
    lambda_D =  12.5 + np.finfo(dtype=np.float16).eps#/lambda_abs + np.finfo(dtype=np.float16).eps  # we weight the dark zone differently w.r.t. the bright zone

    PLOT_RESULTS = False
    only_bright = True
    overfit = False
    bn = False  # batch_normalization
    if control_points:
        if only_bright:
            if overfit & bn:
                saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(
                    N_lspks) + '_cp_' + str(len(c_points_x_) * len(c_points_y_)) + '_lambda' + str(
                    lambda_abs) + '_lr' + str(lr) + '_B' + str(1 / lambda_D) + '_only_bright_decay_overfit32_bn'
            else:
                if switch:
                    saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(N_lspks)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay_switch'
                elif expanded:
                    saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(N_lspks)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay_expanded_fs15_TEST_INV'
                else:
                    saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(N_lspks)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay'
        else:
            if bn:
                saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(N_lspks)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'bn'
            else:
                saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(N_lspks)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)

    elif eval_points:
        if local:
            saved_model_path = 'C:/Users/rales/OneDrive/Desktop/POLIMI/TESI/pressure_matching_deep_learning/models/linear_array/model_linear_config_nl_' + str(
                N_lspks)
        else:
            saved_model_path = '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_'+str(N_lspks)

    log_name = 'lr_'+str(lr)+'_lambaAbs'+str(lambda_abs)+'_lambdaDark'+str(lambda_D)
    if switch:
        log_name += "_switch"
    if expanded:
        log_name+="_expanded"

    # Tensorboard and logging
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # Training params
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)#,clipnorm=100)
    ## CAPIRE COME USARE TENSORBOARD, NELL'ESEMPIO APPLICAVANO SEMPLICEMENTE LA CALLBACK AL COMPILE
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if switch:
        log_dir += "_switch"
    if expanded:
        log_name+="_expanded"
    epoch_to_plot = 25  # Plot evey 'epoch_to_plot' epochs
    val_perc = 0.2  # percentage to separate training and validation set
    early_stop_patience = 100

    # Load configuration
    # Load Green function

    G = np.load(green_function_path)  # green function
    G_B = np.load(green_function_path_bright)  # green function bright zone
    G_D = np.load(green_function_path_dark)  # green function dark zone

    P_gt_ = np.load(gt_soundfield_dataset_path)  # gt soundfield
    P_gt__B = np.load(gt_soundfield_dataset_path_bright)  # gt soundfield bright zone
    P_gt__D = np.load(gt_soundfield_dataset_path_dark)  # gt soundfield dark zone



    if control_points:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(c_points_y_)) * int(len(c_points_x_)), params_linear_2D.N_freqs), dtype=complex)
        P_gt_B = np.zeros((len(params_linear_2D.src_pos_trainT), int(np.floor(len(mingrid_B[0][0])/2)) * int(np.floor(len(mingrid_B[1])/2)), params_linear_2D.N_freqs), dtype=complex)
        P_gt_D = np.zeros((len(params_linear_2D.src_pos_trainT), int(np.floor(len(mingrid_D[0][0])/2)) * int(np.floor(len(mingrid_D[1])/2)), params_linear_2D.N_freqs), dtype=complex)
    elif eval_points:
        P_gt = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(params_linear_2D.idx_lr_gd_y)) * int(len(params_linear_2D.idx_lr_gd_x)), params_linear_2D.N_freqs), dtype=complex)

    normalization_point = 4 * np.pi  # may be useful for future plots if we use sfs

    if control_points:
        sf_shape_x = int(len(c_points_x_)*2)  # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
        sf_shape_y = int(len(c_points_y_)*2/2) # * 2 PERCHé POI CONCATENIAMO PARTE REALE ED IMMAGINARIA
        if only_bright:
            #c_points_y = c_points_y[:int(len(c_points_y)/2)]
            sf_shape = int((sf_shape_x * sf_shape_y) / 2)
        else:
            sf_shape = sf_shape_x * sf_shape_y

        G_cp_y = G[c_points_y_]
        G_cp_ = G_cp_y[:, c_points_x_]
        G_cp = np.zeros((int(sf_shape), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)

        G_cp_y_B = G_B[c_points_y_]
        G_cp__B = G_cp_y_B[:, c_points_x_]
        if only_bright:
            G_cp_B = np.zeros((int(sf_shape), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
        else:
            G_cp_B = np.zeros((int(sf_shape/2), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)

        G_cp_y_D = G_D[c_points_y_]
        G_cp__D = G_cp_y_D[:, c_points_x_]
        if only_bright:
            G_cp_D = np.zeros((int(sf_shape), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)
        else:
            G_cp_D = np.zeros((int(sf_shape/2), params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)

        for i in range(len(params_linear_2D.src_pos_trainT)):
            for j in range(params_linear_2D.N_freqs):
                #P_to_ravel = P_gt_[i, :, :, j]
                P_to_ravel_B = P_gt__B[i, 0:-1:2, 0:-1:2, j]
                P_to_ravel_D = P_gt__D[i, 0:-1:2, 0:-1:2, j]

                #P_gt[i, :, j] = np.ravel(P_to_ravel)
                P_gt_B[i, :, j] = np.ravel(P_to_ravel_B)
                P_gt_D[i, :, j] = np.ravel(P_to_ravel_D)

    _G_ = np.zeros((G.shape[0]**2, params_linear_2D.N_lspks, params_linear_2D.N_freqs), dtype=complex)

    for i in range(params_linear_2D.N_lspks):
        for j in range(params_linear_2D.N_freqs):
            into_G = G[:, :, i, j]
            into_G_cp_B = G_cp__B[:, :, i, j]
            into_G_cp_D = G_cp__D[:, :, i, j]

            _G_[:, i, j] = np.ravel(into_G)
            G_cp_B[:, i, j] = np.ravel(into_G_cp_B)
# ValueError: could not broadcast input array from shape (169,) into shape (338,)
            G_cp_D[:, i, j] = np.ravel(into_G_cp_D)

    #G_cp = tf.convert_to_tensor(G_cp)
    G_ = tf.convert_to_tensor(_G_)
    G_cp_B = tf.convert_to_tensor(G_cp_B)
    G_cp_D = tf.convert_to_tensor(G_cp_D)

    P_train, P_val, src_train, src_val = train_test_split(P_gt_B, params_linear_2D.src_pos_trainT, test_size=val_perc)

    def concat_real_imag(P_, src):
        P_concat_x = tf.concat([tf.math.real(P_), tf.math.imag(P_)], axis=0)
        #P_concat_x_y = tf.concat([tf.math.real(P_concat_x), tf.math.imag(P_concat_x)], axis=1)
        #print("pre-concat = {}, post-1st-concat = {}, post-2nd-concat = {}".format(np.shape(P_), np.shape(P_concat_x), np.shape(P_concat_x_y)))
        return P_concat_x, P_, src

    #def concat_abs_angle(P_, src):
        #P_concat_x = tf.concat([tf.math.divide(tf.math.abs(P_),tf.reduce_max(tf.math.abs(P_))), tf.math.divide(tf.math.angle(P_),tf.reduce_max(tf.math.angle(P_)))], axis=0) # provare a normalizzare
        ##P_concat_x_y = tf.concat([tf.math.real(P_concat_x), tf.math.imag(P_concat_x)], axis=1)
        ##print("pre-concat = {}, post-1st-concat = {}, post-2nd-concat = {}".format(np.shape(P_), np.shape(P_concat_x), np.shape(P_concat_x_y)))
        #return P_concat_x, P_, src

    def concat_abs_angle(P_, src):
        P_concat_x = tf.concat([tf.math.abs(P_), tf.math.angle(P_)], axis=0) # provare a normalizzare
        #P_concat_x_y = tf.concat([tf.math.real(P_concat_x), tf.math.imag(P_concat_x)], axis=1)
        #print("pre-concat = {}, post-1st-concat = {}, post-2nd-concat = {}".format(np.shape(P_), np.shape(P_concat_x), np.shape(P_concat_x_y)))
        return P_concat_x, P_, src

    def preprocess_dataset(P, src):
        data_ds = tf.data.Dataset.from_tensor_slices((P, src))
        preprocessed_ds = data_ds.map(concat_real_imag)
        return preprocessed_ds

    train_ds = preprocess_dataset(P_train, src_train)
    val_ds = preprocess_dataset(P_val, src_val)

    # Define our loss
    loss_fn = tf.keras.losses.MeanAbsoluteError()
    loss_fn_s = tf.keras.losses.MeanSquaredError()  # not used, but improve results
    loss_fn_s_l = tf.keras.losses.MeanSquaredLogarithmicError()  # not used, but improve results

    # Define our metrics
    metric_fn_train_real = tf.keras.metrics.MeanAbsoluteError()
    metric_fn_train_imag = tf.keras.metrics.MeanAbsoluteError()

    metric_fn_val_real = tf.keras.metrics.MeanAbsoluteError()
    metric_fn_val_imag = tf.keras.metrics.MeanAbsoluteError()

    metric_fn_train_dark = tf.keras.metrics.MeanAbsoluteError()
    metric_fn_train_bright = tf.keras.metrics.MeanAbsoluteError()

    metric_fn_val_dark = tf.keras.metrics.MeanAbsoluteError()
    metric_fn_val_bright = tf.keras.metrics.MeanAbsoluteError()

    filter_shape = int(N_lspks)

    N_freqs = params_linear_2D.N_freqs

    # Network Model                                                 idx_cp*2, N_freqs, lspk*2,
    if only_bright:
        network_model_filters = network_utils.pressure_matching_network(2*sf_shape, N_freqs, filter_shape)  # <class 'keras.engine.functional.Functional'>

    else:
        network_model_filters = network_utils.pressure_matching_network(sf_shape, N_freqs, filter_shape)  # <class 'keras.engine.functional.Functional'>

    network_model_filters.summary()
# ValueError: Input 0 of layer "model" is incompatible with the layer: expected shape=(None, 169, 64, 1), found shape=(32, 1352, 64)
    # Load Data
    train_ds = train_ds.shuffle(buffer_size=int(batch_size*2))  # *2))
    val_ds = val_ds.shuffle(buffer_size=int(batch_size*2))  # *2))

    train_ds = train_ds.batch(batch_size=batch_size)
    val_ds = val_ds.batch(batch_size=batch_size)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    exp = False
    if exp:
        saved_model_path+="_exp"
        print("\nPHASE IN EXPONENTIAL\n")


    @tf.function
    def train_step(P_concat, P_):
        with tf.GradientTape() as tape:

            # Compensate driving signals
            d_hat = network_model_filters(P_concat, training=True)[:, :, :, 0]  # <-- Sources/Batch , CPs, Freqs
            d_complex = tf.cast(d_hat[:, :int(d_hat.shape[1] / 2)], dtype=tf.complex64) + (
                        tf.convert_to_tensor(1j, dtype=tf.complex64) * tf.cast(d_hat[:, int(d_hat.shape[1] / 2):],
                                                                                dtype=tf.complex64))

            ##    Batch, CPs, Freqs    VS     Points, Speakers, Freqs    ##
            p_estB = tf.einsum('bij,kij->bkj', d_complex, tf.cast(G_cp_B, dtype=tf.complex64))
            p_estD = tf.einsum('bij,kij->bkj', d_complex, tf.cast(G_cp_D, dtype=tf.complex64))

            P_B = P_
            P_D = np.zeros(np.shape(p_estD))

            if exp:
                loss_value_P_D = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_D)), train_utils.normalize_tensor(tf.math.abs(p_estD)))) * \
                                tf.exp(loss_fn(train_utils.normalize_tensor(tf.math.angle(P_D)), train_utils.normalize_tensor(tf.math.angle(p_estD))))
                loss_value_P_B = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_B)), train_utils.normalize_tensor(tf.math.abs(p_estB)))) * \
                                tf.exp(loss_fn(train_utils.normalize_tensor(tf.math.angle(P_B)), train_utils.normalize_tensor(tf.math.angle(p_estB))))
            else:
                loss_value_P_D = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_D)), train_utils.normalize_tensor(tf.math.abs(p_estD)))) #+ \
                                #loss_fn(train_utils.normalize_tensor(tf.math.angle(P_D)), train_utils.normalize_tensor(tf.math.angle(p_estD)))
                loss_value_P_B = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_B)), train_utils.normalize_tensor(tf.math.abs(p_estB)))) + \
                                loss_fn(train_utils.normalize_tensor(tf.math.angle(P_B)), train_utils.normalize_tensor(tf.math.angle(p_estB)))
            loss_value_P = lambda_D * loss_value_P_D + loss_value_P_B

        network_model_filters_grads = tape.gradient(loss_value_P, network_model_filters.trainable_weights)

        optimizer.apply_gradients(zip(network_model_filters_grads, network_model_filters.trainable_weights))


        if only_bright: # fix here "only bright" otherwise it will have no sense the loss on tensorboard
            metric_fn_train_real.update_state(train_utils.normalize_tensor(tf.concat([tf.math.abs(P_B), tf.math.abs(P_D)], axis=0)),
                                            train_utils.normalize_tensor(tf.concat([tf.math.abs(p_estB), tf.math.abs(p_estD)], axis=0)))
            metric_fn_train_imag.update_state(train_utils.normalize_tensor(tf.math.angle(P_B)),
                                            train_utils.normalize_tensor(tf.math.angle(p_estB)))

            #metric_fn_train_dark.update_state(train_utils.normalize_tensor(tf.math.abs(P_)),train_utils.normalize_tensor(tf.math.abs(p_est[:, :int(p_est.shape[1] / 2), :])))
            #metric_fn_train_bright.update_state(train_utils.normalize_tensor(tf.math.angle(P_)),train_utils.normalize_tensor(tf.math.angle(p_est[:, int(p_est.shape[1] / 2):, :])))
        else:
            metric_fn_train_real.update_state(train_utils.normalize_tensor(tf.math.abs(P_)), train_utils.normalize_tensor(tf.math.abs(p_estB)))
            metric_fn_train_imag.update_state(train_utils.normalize_tensor(tf.math.angle(P_)), train_utils.normalize_tensor(tf.math.angle(p_estB)))

            metric_fn_train_dark.update_state(train_utils.normalize_tensor(tf.math.abs(P_[:, :int(P_.shape[1] / 2 - 6), :])),train_utils.normalize_tensor(tf.math.abs(p_estB[:, :int(P_.shape[1] / 2 - 6), :])))
            metric_fn_train_bright.update_state(train_utils.normalize_tensor(tf.math.angle(P_[:, int(P_.shape[1] / 2 - 6):, :])),train_utils.normalize_tensor(tf.math.angle(p_estB[:, int(P_.shape[1] / 2 - 6):, :])))

        return loss_value_P

    @tf.function
    def val_step(P_concat, P_):

        # Compensate driving signals
        d_hat = network_model_filters(P_concat, training=False)[:, :, :, 0]

        d_complex = tf.cast(d_hat[:, :int(d_hat.shape[1] / 2)], dtype=tf.complex64) + (
                tf.convert_to_tensor(1j, dtype=tf.complex64) * tf.cast(d_hat[:, int(d_hat.shape[1] / 2):],
                                                                        dtype=tf.complex64))

        ##    Batch, CPs, Freqs    VS     Points, Speakers, Freqs    ##
        p_estB = tf.einsum('bij,kij->bkj', d_complex, tf.cast(G_cp_B, dtype=tf.complex64))
        p_estD = tf.einsum('bij,kij->bkj', d_complex, tf.cast(G_cp_D, dtype=tf.complex64))


        P_B = P_
        P_D = np.zeros(np.shape(p_estD))

        # loss_value_P_abs = loss_fn(train_utils.normalize_tensor(tf.math.abs(P_)), train_utils.normalize_tensor(tf.math.abs(p_est)))
        # loss_value_P_angle = loss_fn(train_utils.normalize_tensor(tf.math.angle(P_)), train_utils.normalize_tensor(tf.math.angle(p_est)))
        # lambda_abs = loss_value_P_angle / loss_value_P_abs
        # loss_value_P = (lambda_abs * loss_value_P_abs + loss_value_P_angle)

        if exp:
            loss_value_P_D = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_D)),
                                                   train_utils.normalize_tensor(tf.math.abs(p_estD)))) * \
                             tf.exp(loss_fn(train_utils.normalize_tensor(tf.math.angle(P_D)),
                                            train_utils.normalize_tensor(tf.math.angle(p_estD))))
            loss_value_P_B = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_B)),
                                                   train_utils.normalize_tensor(tf.math.abs(p_estB)))) * \
                             tf.exp(loss_fn(train_utils.normalize_tensor(tf.math.angle(P_B)),
                                            train_utils.normalize_tensor(tf.math.angle(p_estB))))
        else:
            loss_value_P_D = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_D)),
                                                   train_utils.normalize_tensor(tf.math.abs(p_estD)))) #+ \
                             #loss_fn(train_utils.normalize_tensor(tf.math.angle(P_D)),
                                     #train_utils.normalize_tensor(tf.math.angle(p_estD)))
            loss_value_P_B = (lambda_abs * loss_fn(train_utils.normalize_tensor(tf.math.abs(P_B)),
                                                   train_utils.normalize_tensor(tf.math.abs(p_estB)))) + \
                             loss_fn(train_utils.normalize_tensor(tf.math.angle(P_B)),
                                     train_utils.normalize_tensor(tf.math.angle(p_estB)))
        loss_value_P = lambda_D * loss_value_P_D + loss_value_P_B

        if only_bright:  # fix here "only bright" otherwise it will have no sense the loss on tensorboard
            metric_fn_val_real.update_state(train_utils.normalize_tensor(tf.concat([tf.math.abs(P_B), tf.math.abs(P_D)], axis=0)),
                                            train_utils.normalize_tensor(tf.concat([tf.math.abs(p_estB), tf.math.abs(p_estD)], axis=0)))
            metric_fn_val_imag.update_state(train_utils.normalize_tensor(tf.math.angle(P_B)),
                                            train_utils.normalize_tensor(tf.math.angle(p_estB)))

            # metric_fn_val_dark.update_state(train_utils.normalize_tensor(tf.math.abs(P_)),
            # train_utils.normalize_tensor(
            # tf.math.abs(p_est[:, :int(p_est.shape[1] / 2), :])))
            # metric_fn_val_bright.update_state(train_utils.normalize_tensor(tf.math.angle(P_)),
            # train_utils.normalize_tensor(
            # tf.math.angle(p_est[:, int(p_est.shape[1] / 2):, :])))
        else:
            metric_fn_val_real.update_state(train_utils.normalize_tensor(tf.math.abs(P_)),
                                            train_utils.normalize_tensor(tf.math.abs(p_estB)))
            metric_fn_val_imag.update_state(train_utils.normalize_tensor(tf.math.angle(P_)),
                                            train_utils.normalize_tensor(tf.math.angle(p_estB)))

            metric_fn_val_dark.update_state(
                train_utils.normalize_tensor(tf.math.abs(P_[:, :int(P_.shape[1] / 2 - 6), :])),
                train_utils.normalize_tensor(tf.math.abs(p_estB[:, :int(P_.shape[1] / 2 - 6), :])))
            metric_fn_val_bright.update_state(
                train_utils.normalize_tensor(tf.math.angle(P_[:, int(P_.shape[1] / 2 - 6):, :])),
                train_utils.normalize_tensor(tf.math.angle(p_estB[:, int(P_.shape[1] / 2 - 6):, :])))

        return loss_value_P, d_hat, p_estB, p_estD, d_complex

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'+'_nl'+str(
            N_lspks)+'_xmin'+str(params_linear_2D.x_min_train)+'_cp'+str(sf_shape)+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'D_decay'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'+'_nl'+str(
            N_lspks)+'_xmin'+str(params_linear_2D.x_min_train)+'_cp'+str(sf_shape)+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'D_decay'
    if overfit:
        train_log_dir = train_log_dir+'_overfit'
        test_log_dir = test_log_dir+'_overfit'
    if bn:
        train_log_dir = train_log_dir+'_bn'
        test_log_dir = test_log_dir+'_bn'
    if switch:
        train_log_dir = train_log_dir+'_switch'
        test_log_dir = test_log_dir+'_switch'

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for n_e in tqdm(range(epochs)):

        plot_val = True
        for P_concat, P, _ in train_ds:
            loss_value_P = train_step(P_concat, P)

        ## For Tensorboard
        train_loss_A = metric_fn_train_real.result()
        train_loss_P = metric_fn_train_imag.result()
        train_loss__ = lambda_abs * train_loss_A + train_loss_P

        #train_loss_ = metric_fn_train_dark.result() + metric_fn_train_bright.result()

        # Log to tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss_A', train_loss_A, step=n_e)
            tf.summary.scalar('train_loss_P', train_loss_P, step=n_e)
            tf.summary.scalar('train_loss__', train_loss__, step=n_e)

        for P_concat, P, src  in val_ds:
            loss_value_P_val, d_hat, p_estB, p_estD, d_complex = val_step(P_concat, P)

        val_loss_A = metric_fn_val_real.result()
        val_loss_P = metric_fn_val_imag.result()
        val_loss__ = lambda_abs * val_loss_A + val_loss_P

        #val_loss_ = metric_fn_val_bright.result() + metric_fn_val_dark.result()

        with test_summary_writer.as_default():
            tf.summary.scalar('val_loss_A', val_loss_A, step=n_e)
            tf.summary.scalar('val_loss_P', val_loss_P, step=n_e)
            tf.summary.scalar('val_loss__', val_loss__, step=n_e)

        template_real = 'Epoch {}, Real-Loss_train: {}, Real-Loss_val: {}'
        template_imag = 'Epoch {}, Imag-Loss_train: {}, Imag-Loss_val: {}'
        #template_bright = 'Bright-Loss_train: {}, Bright-Loss_val: {}'
        #template_dark = 'Dark-Loss_train: {}, Dark-Loss_val: {}'
        print(template_real.format(n_e + 1, metric_fn_train_real.result(), metric_fn_val_real.result()))  # QUESTO DOVREBBE ESSERE SCOMMENTATO ?
        print(template_imag.format(n_e + 1, metric_fn_train_imag.result(), metric_fn_val_imag.result()))
        #print(template_bright.format(n_e + 1, metric_fn_train_bright.result(), metric_fn_val_bright.result()))
        #print(template_dark.format(n_e + 1, metric_fn_train_dark.result(), metric_fn_val_dark.result()))

        # Reset metrics every epoch
        metric_fn_train_real.reset_states()
        metric_fn_train_imag.reset_states()
        metric_fn_val_real.reset_states()
        metric_fn_val_imag.reset_states()

        #metric_fn_train_dark.reset_states()
        #metric_fn_train_bright.reset_states()
        #metric_fn_val_dark.reset_states()
        #metric_fn_val_bright.reset_states()
        #train_loss.reset_states() #?

        # %tensorboard --logdir logs/gradient_tape --port 6000  # da terminale, mettendo la porta # senza % sulla linea di comando
        # --port  # mi connetto dal mio computer tramite ssh alla porta --> ssh -L 16000:127.0.0.1:6000 ralessandri@10.79.0.8 nsurname@my_server_ip
        # dal browser localhost:port --> http://127.0.0.1:16000/

        #!tensorboard dev upload \
            #--logdir logs/fit \
            #--name "(optional) My latest experiment" \
            #--description "(optional) Simple comparison of several hyperparameters" \
            #--one_shot

        # Every epoch_to_plot epochs plot an example of validation
        if  not n_e % epoch_to_plot and plot_val:
            print('Train loss: ' + str(train_loss__.numpy()))
            print('Val loss: ' + str(val_loss__.numpy()))
            #print('Train loss_B-D: ' + str(train_loss_.numpy()))
            #print('Val loss_B-D: ' + str(val_loss_.numpy()))

            n_f = np.random.randint(1, 63) #41 #int(len(params_linear_2D.f_axis) * len(params_linear_2D.f_axis) / (4 * params_linear_2D.N_lspks)  - 1) # FIXED to 900 Hz

            index = np.random.randint(0, p_estB.shape[0]) # showing always a different source
            P_D = np.zeros(p_estB.shape[1])

            if control_points:
                #p_est = np.sum(G * d_complex, axis=0)# tf.einsum('bij,kij->bkj', d_complex, tf.cast(G_, dtype=tf.complex64))
                p_est = tf.einsum('bij,kij->bkj', d_complex, tf.cast(G_, dtype=tf.complex64))
                #p_est_concatenated = np.concatenate((p_estB[index, :, n_f], p_estD[index, :, n_f]))
                p_est_reshapedB = np.reshape(p_estB[index, :, n_f], (int(len(c_points_x_)), int(len(c_points_y_))))
                p_est_reshapedD = np.reshape(p_estD[index, :, n_f], (int(len(c_points_x_)), int(len(c_points_y_))))
                p_est_reshaped = np.reshape(p_est[index, :, n_f], (G.shape[0], G.shape[0]))

                p_to_gt = P.numpy()[index, :, n_f]
                p_to_gt_ = np.concatenate((p_to_gt, P_D))
                p_gt = np.reshape(p_to_gt, (int(len(c_points_x_)), int(len(c_points_y_))))
                #p_gt = np.reshape(np.real(P.numpy()[index, :, n_f].concat(P_D)), (int(len(c_points_y)), int(len(c_points_x))))

            limit = np.max(np.abs(p_gt))
            figure_soundfield = plt.figure(figsize=(10, 20))
            plt.subplot(411)
            plt.imshow(np.real(p_gt), cmap='coolwarm')
            plt.title('GT, Point Source at {} m and frequency {}'.format(params_linear_2D.src_pos_trainT[index], params_linear_2D.f_axis[n_f]))
            plt.clim(-limit, limit)
            plt.colorbar()
            plt.subplot(412)
            plt.imshow(np.real(p_est_reshapedB), cmap='coolwarm')
            plt.title('Estimate of Bright Zone')
            plt.clim(-limit, limit)
            plt.colorbar()
            plt.subplot(413)
            plt.imshow(np.real(p_est_reshapedD), cmap='coolwarm')
            plt.title('Estimate of Dark Zone')
            plt.clim(-limit, limit)
            plt.colorbar()
            plt.subplot(414)
            plt.imshow(np.real(p_est_reshaped), cmap='coolwarm')
            plt.title('Estimate Whole field')
            plt.clim(-limit, limit)
            plt.colorbar()
            plt.show()

            with test_summary_writer.as_default():
                tf.summary.image("soundfield second training", train_utils.plot_to_image(figure_soundfield),
                                    step=n_e)

            filters_fig = plt.figure()
            plt.plot(d_hat.numpy()[0, :, :])
            with train_summary_writer.as_default():
                tf.summary.image("Filters true second", train_utils.plot_to_image(filters_fig), step=n_e)

                ## ##

                # Select best model
        if n_e == 0:
            lowest_val_loss = val_loss__
            network_model_filters.save(saved_model_path+'reIm')
            early_stop_counter = 0

        else:
            if val_loss__ < lowest_val_loss:
                network_model_filters.save(saved_model_path+'reIm')
                lowest_val_loss = val_loss__
                early_stop_counter = 0
                best_epoch = n_e
            else:
                # network_model_filters.save(saved_model_path)
                early_stop_counter = early_stop_counter + 1

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print('Training finished at epoch ' + str(n_e))
            print("actual val loss = {}\nbest val loss = {} @ the {}th epoch".format(val_loss__, lowest_val_loss,
                                                                                         best_epoch))
            break

        # Log to tensorboard
        with summary_writer.as_default():
            tf.summary.scalar('val_loss_P', val_loss__, step=n_e)
            tf.summary.scalar('val_abs', metric_fn_val_real.result(), step=n_e)
            tf.summary.scalar('val_phase', metric_fn_val_imag.result(), step=n_e)

        metric_fn_val_real.reset_states()
        metric_fn_val_imag.reset_states()

        print('model at \n'+saved_model_path+'reIm')

    return

if __name__ == '__main__':
    main()
    print("End Training in Main")