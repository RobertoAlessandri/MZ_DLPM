import tensorflow as tf
from data_lib import params_linear_2D
 # filter_shape, sf_shape_x, nfft



def pressure_matching_network(sf_shape_x, nfft, filter_shape):

    # Encoder  # sf_shape_x, nfft, 1
    input_layer = tf.keras.layers.Input(shape=(sf_shape_x, nfft, 1))  # 300, 64, 1
    x = tf.keras.layers.Conv2D(32, 3, 2, padding='same', kernel_regularizer='l2')(input_layer)  # 150, 32, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, 1, padding='same')(x)  # 150, 32, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, padding='same', kernel_regularizer='l2')(x)  # 150, 32, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(x)  # 75, 16, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, padding='same', kernel_regularizer='l2')(x)  # 38, 8, 128
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(x)  # 38, 8, 128
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(256, 3, 2, padding='same', kernel_regularizer='l2')(x)  # 10, 2, 256
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(x)  # 10, 2, 256
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2D(512, 3, 2, padding='same', kernel_regularizer='l2')(x)  # 5, 1, 512
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, 3, 1, padding='same')(x)  # 5, 1, 512
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Flatten()(x)  # 1280
        #x = tf.keras.layers.Dropout(0.8)(x)

    # Bottleneck
    depth_decoder = 4
    bottleneck_dim_filters = int(params_linear_2D.N_lspks / 16) # 4 if 16, 8 if 32, 16 if 64, 32 if 128
    bottleneck_dim_freq = 2
    x = tf.keras.layers.Dense(bottleneck_dim_filters*bottleneck_dim_freq, activity_regularizer='l1_l2')(x)  # 128
        #x = tf.keras.layers.PReLU()(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        #x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Reshape((bottleneck_dim_filters, bottleneck_dim_freq,1))(x)  # 16, 8, 1

    # Decoder
    x = tf.keras.layers.Conv2DTranspose(512, 3, 2, padding='same')(x)  # 32, 16, 256
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(512, 3, 1, padding='same')(x)  # 32, 16, 256
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same')(x)  # 32, 16, 256
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(256, 3, 1, padding='same')(x)  # 32, 16, 256
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same')(x)  # 64, 32, 128
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, 1, padding='same')(x)  # 64, 32, 128
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same')(x)  # 128, 64, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, 1, padding='same')(x)  # 128, 64, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same')(x)  # 128, 64, 64
    x = tf.keras.layers.PReLU()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, 1, padding='same')(x)  # 128, 64, 64
    x = tf.keras.layers.PReLU()(x)

    # Output
    x_out = tf.keras.layers.Conv2DTranspose(1, 3, 1, padding='same')(x)# activation="sigmoid")(x)  # 128, 64, 1
    ##x_out = tf.keras.layers.PReLU()(x_out)

    ##x_out = tf.keras.layers.Dense(128)(x_out)  # 128
    ##x_out = tf.keras.layers.PReLU()(x_out)
        #x_out = tf.keras.layers.Dropout(0.8)(x_out)
    ##x_out = tf.keras.layers.Dense(64)(x_out)  # 128
    ##x_out = tf.keras.layers.PReLU()(x_out)
    ##x_out = tf.keras.layers.BatchNormalization()(x_out)
    ##x_out = tf.keras.layers.Dense(32, kernel_regularizer='l2')(x_out)  # 128
    ##x_out = tf.keras.layers.Dense(16, activity_regularizer='l1_l2')(x_out)  # 128
        #x_out = tf.keras.layers.Dropout(0.5)(x_out)
    ##x_out = tf.keras.layers.Dense(8, bias_regularizer='l1')(x_out)  # 128
        #x_out = tf.keras.layers.Dropout(0.2)(x_out)
    ##x_out = tf.keras.layers.Dense(4)(x_out)  # 128
    ##x_out = tf.keras.layers.Dense(2)(x_out)  # 128
    ##x_out = tf.keras.layers.Dense(1)(x_out)  # 128

        #x_out = tf.keras.layers.Dropout(0.2)(x_out)
        #out = tf.keras.layers.Reshape((bottleneck_dim_filters* 8, bottleneck_dim_freq * 8 ,1))(x_out_)  # 128, 64, 1


    return tf.keras.models.Model(inputs=input_layer, outputs=x_out)
print("Network_Utils ended")