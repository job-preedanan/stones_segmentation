import tensorflow as tf


class MultiResUnet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256

        # encoder parameters
        self.CONV_STRIDE = 1
        self.MAX_POOLING_SIZE = 2
        self.MAX_POOLING_STRIDE = 2

        # decoder parameters
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.CONCATENATE_AXIS = -1

        # encoder
        inputs = tf.keras.Input(shape=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        multi_res_block1 = self._add_multi_res_block(first_filter_count=16, input_sequence=inputs)
        res_path1 = self._add_res_path_block(filter_count=32, conv_count=4, input_sequence=multi_res_block1)
        multi_res_block1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(multi_res_block1)

        multi_res_block2 = self._add_multi_res_block(first_filter_count=32, input_sequence=multi_res_block1)
        res_path2 = self._add_res_path_block(filter_count=64, conv_count=3, input_sequence=multi_res_block2)
        multi_res_block2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(multi_res_block2)

        multi_res_block3 = self._add_multi_res_block(first_filter_count=64, input_sequence=multi_res_block2)
        res_path3 = self._add_res_path_block(filter_count=128, conv_count=2, input_sequence=multi_res_block3)
        multi_res_block3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(multi_res_block3)

        multi_res_block4 = self._add_multi_res_block(first_filter_count=128, input_sequence=multi_res_block3)
        res_path4 = self._add_res_path_block(filter_count=256, conv_count=1, input_sequence=multi_res_block4)
        multi_res_block4 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(multi_res_block4)

        multi_res_block5 = self._add_multi_res_block(first_filter_count=256, input_sequence=multi_res_block4)

        # decoder
        multi_res_block6 = self._add_upsampling_layer(filter_count=128, input_sequence=multi_res_block5)
        multi_res_block6 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block6, res_path4])
        multi_res_block6 = self._add_multi_res_block(first_filter_count=128, input_sequence=multi_res_block6)

        multi_res_block7 = self._add_upsampling_layer(filter_count=128, input_sequence=multi_res_block6)
        multi_res_block7 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block7, res_path3])
        multi_res_block7 = self._add_multi_res_block(first_filter_count=64, input_sequence=multi_res_block7)

        multi_res_block8 = self._add_upsampling_layer(filter_count=64, input_sequence=multi_res_block7)
        multi_res_block8 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block8, res_path2])
        multi_res_block8 = self._add_multi_res_block(first_filter_count=32, input_sequence=multi_res_block8)

        multi_res_block9 = self._add_upsampling_layer(filter_count=32, input_sequence=multi_res_block8)
        multi_res_block9 = tf.keras.layers.Concatenate(axis=-1)([multi_res_block9, res_path1])
        multi_res_block9 = self._add_multi_res_block(first_filter_count=16, input_sequence=multi_res_block9)

        # additional upsampling layer for vgg loss
        # multi_res_block10 = UpSampling2D()(multi_res_block9)   # simple upsampling

        # output
        outputs = tf.keras.layers.Conv2D(filters=1,
                                         kernel_size=1,
                                         strides=1,
                                         padding='same')(multi_res_block9)
        outputs = tf.keras.layers.BatchNormalization()(outputs)
        outputs = tf.keras.layers.Activation(activation='sigmoid')(outputs)

        # stack to make 3 layers output
        # outputs = tf.keras.layers.Concatenate([outputs, outputs, outputs], axis=-1)
        # temp = concatenate([outputs, outputs], axis=-1)
        # outputs = concatenate([temp, outputs], axis=-1)

        self.MultiResUnet = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.MultiResUnet.summary()

    def _add_multi_res_block(self, first_filter_count, input_sequence):

        # 3 convolution layers
        layer1 = self._add_convolution_layer(first_filter_count, input_sequence, kernel_size=3, activation='relu')
        layer2 = self._add_convolution_layer(first_filter_count*2, layer1, kernel_size=3, activation='relu')
        layer3 = self._add_convolution_layer(first_filter_count*4, layer2, kernel_size=3, activation='relu')

        # concatenate
        concatenate_layer = tf.keras.layers.Concatenate(axis=-1)([layer3, layer2])
        concatenate_layer = tf.keras.layers.Concatenate(axis=-1)([concatenate_layer, layer1])

        # add with residual layer
        total_filter_count = first_filter_count + first_filter_count*2 + first_filter_count*4
        shortcut = self._add_convolution_layer(total_filter_count, input_sequence, kernel_size=1, activation='relu')
        new_sequence = tf.keras.layers.add([shortcut, concatenate_layer])

        return new_sequence

    def _add_res_path_block(self, filter_count, conv_count, input_sequence):

        for n in range(conv_count):
            new_sequence = self._add_convolution_layer(filter_count, input_sequence, kernel_size=3, activation='relu')
            shortcut = self._add_convolution_layer(filter_count, input_sequence, kernel_size=1, activation='relu')
            new_sequence = tf.keras.layers.add([shortcut, new_sequence])

        return new_sequence

    def _add_convolution_layer(self, filter_count, sequence, kernel_size, activation):
        sequence = tf.keras.layers.Conv2D(filter_count,
                                          kernel_size=kernel_size,
                                          strides=self.CONV_STRIDE,
                                          padding='same')(sequence)
        sequence = tf.keras.layers.BatchNormalization()(sequence)
        if activation == 'relu':
            new_sequence = tf.keras.layers.ReLU()(sequence)
        elif activation == 'leaky_relu':
            new_sequence = tf.keras.layers.LeakyReLU(0.2)(sequence)

        return new_sequence

    def _add_upsampling_layer(self, filter_count, input_sequence):
        new_sequence = tf.keras.layers.Conv2DTranspose(filter_count,
                                                       kernel_size=self.DECONV_FILTER_SIZE,
                                                       strides=self.DECONV_STRIDE,
                                                       kernel_initializer='he_uniform')(input_sequence)
        new_sequence = tf.keras.layers.BatchNormalization()(new_sequence)

        return new_sequence

    def get_model(self):
        return self.MultiResUnet
