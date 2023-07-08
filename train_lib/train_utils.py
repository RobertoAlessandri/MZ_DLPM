import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize(x):
    x_mean = np.mean(x)
    x_std_dev = np.sqrt(np.var(x))
    x_norm = (x - x_mean) / x_std_dev
    x_norm = x
    return x_norm, x_mean, x_std_dev


def normalize_graph(x):
    x_mean = tf.math.reduce_mean(x)
    x_std_dev = tf.math.sqrt(tf.math.reduce_variance(x))
    x_norm = (x - x_mean) / x_std_dev
    #x_norm = x
    return x_norm #, x_mean, x_std_dev


def denormalize(x_norm, x_mean, x_std_dev):
    x_denorm = (x_norm * x_std_dev) + x_mean
    x_denorm = x_norm
    return x_denorm

def normalize_tensor(input_tensor, do_normalize=False):
    """ Normalizes 3D tensor between 0 and 1
    Args:
        input_tensor, tf.Tensor
    Returns:
        norm_tensor, tf.Tensor
    """
    max_val = tf.math.reduce_max(tf.reshape(input_tensor, shape=[input_tensor.shape[0], -1])) #), axis=-1)
    min_val = tf.math.reduce_min(tf.reshape(input_tensor, shape=[input_tensor.shape[0], -1])) #, axis=-1)
    # norm_tensor = tf.divide(input_tensor - tf.expand_dims(tf.expand_dims(tf.expand_dims(min_val, 1), 2),3),
    #                        tf.expand_dims(tf.expand_dims(tf.expand_dims(max_val - min_val, 1), 2),3))
    if do_normalize:
        return tf.divide(input_tensor - min_val, max_val - min_val)
    else:
        return input_tensor

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def est_vs_gt_filt_soundfield_fig(h_hat, h, h_min_val, h_max_val, SRG_inst, wc):

    example_idx = tf.random.uniform(
        shape=[1], minval=0, maxval=h_hat.shape[0], dtype=tf.int32)

    h_hat_example = denormalize(h_hat.numpy()[example_idx], h_min_val, h_max_val)
    h_example = denormalize(h.numpy()[example_idx], h_min_val, h_max_val)

    # Need to reshape coeff in complex vector
    h_hat_example = h_hat_example[:int(h_hat_example.shape[0] / 2)] + 1j * h_hat_example[
                                                                           int(h_hat_example.shape[0] / 2):]
    h_example = h_example[:int(h_example.shape[0] / 2)] + 1j * h_example[int(h_example.shape[0] / 2):]

    P_est_val = SRG_inst.estimate_sounfield(wc, h_hat_example)
    P_gt_val = SRG_inst.estimate_sounfield(wc, h_example)

    figure_filt = plt.figure()
    plt.subplot(221)
    plt.title('Sounfield Re{} Est filters')
    plt.imshow(np.real(P_est_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.subplot(222)
    plt.title('Sounfield Im{} Est filters')
    plt.imshow(np.imag(P_est_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.subplot(223)
    plt.title('Sounfield Re{} Gt filters')
    plt.imshow(np.real(P_gt_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.subplot(224)
    plt.title('Sounfield Im{} Gt filters')
    plt.imshow(np.imag(P_gt_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.tight_layout()

    return figure_filt


def est_vs_gt_soundfield(P_hat, P_gt):

    # example_idx = tf.random.uniform(
    #    shape=[1], minval=0, maxval=P_hat.shape[0], dtype=tf.int32)

    example_idx = np.random.randint(low=0, high=P_hat.shape[0])


    figure_sf = plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title('Est soundfield')
    plt.imshow(np.real(P_hat.numpy()[example_idx, :, :, 0]), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.subplot(122)
    plt.title('GT soundfield')
    plt.imshow(np.real(P_gt.numpy()[example_idx, :, :, 0]), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.tight_layout()

    return figure_sf


def est_vs_gt_filt_soundfield_fig_train_2(h_hat, h, h_min_val, h_max_val, SRG_inst, wc, mask):

    # example_idx = tf.random.uniform(
    #    shape=[1], minval=0, maxval=h_hat.shape[0], dtype=tf.int32)

    example_idx = np.random.randint(low=0, high=h_hat.shape[0])


    h_hat_example = denormalize(h_hat.numpy()[example_idx], h_min_val, h_max_val)
    h_example = denormalize(h.numpy()[example_idx], h_min_val, h_max_val)

    h_hat_example_hole = np.multiply(h_hat_example, mask.numpy()[example_idx])
    h_example_hole = np.multiply(h_example, mask.numpy()[example_idx])

    # Need to reshape coeff in complex vector
    h_hat_example_hole = h_hat_example_hole[:int(h_hat_example_hole.shape[0] / 2)] + 1j * h_hat_example_hole[
                                                                           int(h_hat_example_hole.shape[0] / 2):]
    h_example_hole = h_example_hole[:int(h_example_hole.shape[0] / 2)] + 1j * h_example_hole[
                                                                                          int(h_example_hole.shape[
                                                                                                  0] / 2):]
    h_example = h_example[:int(h_example.shape[0] / 2)] + 1j * h_example[int(h_example.shape[0] / 2):]

    P_est_val = SRG_inst.estimate_sounfield(wc, h_hat_example_hole)
    P_gt_val = SRG_inst.estimate_sounfield(wc, h_example)
    P_gt_val_hole = SRG_inst.estimate_sounfield(wc, h_example_hole)

    figure = plt.figure(figsize=(20, 10))
    plt.subplot(231)
    plt.title('Sounfield Re{} Est filters')
    plt.imshow(np.real(P_est_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.tight_layout()
    plt.subplot(232)
    plt.title('Sounfield Re{} Gt filters holes')
    plt.imshow(np.real(P_gt_val_hole), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.tight_layout()
    plt.subplot(233)
    plt.title('Sounfield Re{} Gt filters')
    plt.imshow(np.real(P_gt_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.tight_layout()

    plt.subplot(234)
    plt.title('Sounfield Im{} Est filters')
    plt.imshow(np.imag(P_est_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.subplot(235)
    plt.title('Sounfield Im{} Gt filters holes')
    plt.imshow(np.imag(P_gt_val_hole), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.tight_layout()
    plt.subplot(236)
    plt.title('Sounfield Im{} Gt filters')
    plt.imshow(np.imag(P_gt_val), aspect='auto'), plt.colorbar(), plt.gca().invert_yaxis()
    plt.xlabel('x [m]'), plt.ylabel('y [m]')
    plt.tight_layout()
    return figure
print("Train_Utils ended")