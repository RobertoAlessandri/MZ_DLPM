import tensorflow as tf
from data_lib import params_linear_2D
import numpy as np
import matplotlib.pyplot as plt


def concat_real_imag(P_, src):
    P_concat_x = tf.concat([tf.math.real(P_), tf.math.imag(P_)], axis=0)
    # P_concat_x_y = tf.concat([tf.math.real(P_concat_x), tf.math.imag(P_concat_x)], axis=1)
    # print("pre-concat = {}, post-1st-concat = {}, post-2nd-concat = {}".format(np.shape(P_), np.shape(P_concat_x), np.shape(P_concat_x_y)))
    return P_concat_x, P_, src


def preprocess_dataset(P, src):
    data_ds = tf.data.Dataset.from_tensor_slices((P, src))
    preprocessed_ds = data_ds.map(concat_real_imag)
    return preprocessed_ds



N_lspks = params_linear_2D.N_lspks
c_points_x = params_linear_2D.idx_cp_x2_expanded
c_points_y = params_linear_2D.idx_cp_y2_expanded

lambda_abs = 7.5 + np.finfo(dtype=np.float16).eps # we weight the absolute value loss since it is in a different range w.r.t. phase
lambda_D =  1/lambda_abs + np.finfo(dtype=np.float16).eps
lr = 3*0.001
cp =len(params_linear_2D.idx_cp_x2_expanded * 2)
model = tf.keras.models.load_model(
    '/nas/home/ralessandri/thesis_project/models/linear_array/model_linear_config_nl_' + str(N_lspks)+'_cp_'+str(len(c_points_x)*len(c_points_y))+'_lambda'+str(lambda_abs)+'_lr'+str(lr)+'_B'+str(1/lambda_D)+'_only_bright_decay_expanded_fs15reIm')
gt_soundfield_dataset_path = '/nas/home/ralessandri/thesis_project/dataset/linear_array/bright/gt_soundfield_train' + '_nl' + str(
        params_linear_2D.N_lspks) + '_half_cp'+str(cp)+'_xmin'+str(params_linear_2D.x_min_train)+'_decay_expanded_fs15.npy'
P_gt_ = np.load(gt_soundfield_dataset_path)
# 1500 15 12 64
n_r = 1
n_f = 16
c_points_x = params_linear_2D.idx_cp_x2
c_points_y = params_linear_2D.idx_cp_y2
only_bright = True
if only_bright:
    P_gt__B = np.load(gt_soundfield_dataset_path)
#P_gt = np.zeros(
    #(len(params_linear_2D.src_pos_trainT), int(len(c_points_y)) * int(len(c_points_x)), params_linear_2D.N_freqs),
    #dtype=complex)

#for i in range(len(params_linear_2D.src_pos_trainT)):
    #for j in range(params_linear_2D.N_freqs):
        #P_to_ravel = P_gt_[i, :, :, j]
        #P_gt[i, :, j] = np.ravel(P_to_ravel)
        #if only_bright:
            #P_gt_B[i, :, j] = np.ravel(P_to_ravel)[int(len(np.ravel(P_to_ravel)) / 2):]

mingrid_B = params_linear_2D.mingrid_B_expanded
mingrid_D = params_linear_2D.mingrid_D_expanded

P_gt_B = np.zeros((len(params_linear_2D.src_pos_trainT), int(len(mingrid_B[0][0]) / 2) * int(len(mingrid_B[1]) / 2),
                   params_linear_2D.N_freqs), dtype=complex)
for i in range(len(params_linear_2D.src_pos_trainT)):
    for j in range(params_linear_2D.N_freqs):
        P_to_ravel_B = P_gt__B[i, 0:-1:2, 0:-1:2, j]

        P_gt_B[i, :, j] = np.ravel(P_to_ravel_B)

if only_bright:
    train_ds = preprocess_dataset(P_gt_B, params_linear_2D.src_pos_trainT)
#else:
   #train_ds = preprocess_dataset(P_gt, params_linear_2D.src_pos_trainT)
# inputs=model.inputs
extractor = tf.keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
train_ds = train_ds.batch(1)
for P_concat, P_, _ in train_ds:
    exp = np.expand_dims(P_concat, axis=-1)
    #exp2 = np.expand_dims(exp, axis=0)

    #exp2 = P_concat.reshape(P_concat.shape[0], P_concat.shape[1], 1)
    #exp3 = P_concat.reshape(P_concat.shape[0], P_concat.shape[1], 1)
    #input = (None, exp[0], exp[1], exp[2])
    features = extractor(tf.convert_to_tensor(exp))
for i in range(len(features)):
    if np.ndim(features[i]) == 4:
        plt.figure(figsize=(10, 20))
        #plt.subplot(121)
        plt.imshow(np.mean(np.mean(features[i], axis=-1), axis=0), aspect='equal')
        #plt.subplot(122)
        #plt.imshow(np.mean(np.mean(features[i], axis=-1), axis=0), aspect='auto')
        plt.title('n_features = {}, with shape = {}'.format(i, np.shape(features[i])))
        plt.show()
        #print(np.shape(features[i]))
# bottleneck