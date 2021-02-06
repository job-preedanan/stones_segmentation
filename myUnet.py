from keras.models import Model
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import Input, LeakyReLU, BatchNormalization, Activation, Dropout, MaxPooling2D, Add
from keras.regularizers import l2



class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256

        # encoder parameters
        self.CONV_FILTER_SIZE = 3
        self.CONV_STRIDE = 1
        self.CONV_PADDING = (1, 1)
        self.MAX_POOLING_SIZE = 3
        self.MAX_POOLING_STRIDE = 2
        self.MAX_POOLING_PADDING = (1, 1)

        # decoder parameters
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2
        self.CONCATENATE_AXIS = -1

        # -------------------------------------------ENCODER PART ------------------------------------------------------
        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))
        enc0 = self._add_convolution_block(first_layer_filter_count, inputs, 'ReLU')
        enc0 = self._add_convolution_block(first_layer_filter_count, enc0, 'ReLU')
        
        # (128 x 128 x 2N)
        filter_count = first_layer_filter_count * 2
        enc1 = self._add_encoding_layer(filter_count, enc0)

        # (64 x 64 x 4N)
        filter_count = first_layer_filter_count * 4
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 8N)
        filter_count = first_layer_filter_count * 8
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 16N)
        filter_count = first_layer_filter_count * 16
        enc4 = self._add_encoding_layer(filter_count, enc3)
        enc4 = Dropout(0.5)(enc4)

        # -------------------------------------------DECODER PART ------------------------------------------------------
        # (32 x 32 x 8N)
        filter_count = first_layer_filter_count * 8
        dec3 = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE,
                               strides=self.DECONV_STRIDE,
                               kernel_initializer='he_uniform')(enc4)
        dec3 = BatchNormalization()(dec3)
        dec3 = Dropout(0.5)(dec3)
        dec3 = concatenate([dec3, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 4N)
        filter_count = first_layer_filter_count * 4
        dec2 = self._add_decoding_layer(filter_count, True, dec3)
        dec2 = concatenate([dec2, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)
        filter_count = first_layer_filter_count * 2
        dec1 = self._add_decoding_layer(filter_count, True, dec2)
        dec1 = concatenate([dec1, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x N)
        dec0 = self._add_decoding_layer(first_layer_filter_count, True, dec1)
        dec0 = concatenate([dec0, enc0], axis=self.CONCATENATE_AXIS)

        # Last layer : CONV BLOCK + convolution + Sigmoid
        dec0 = self._add_convolution_block(first_layer_filter_count, dec0, 'ReLU')
        dec0 = self._add_convolution_block(first_layer_filter_count, dec0, 'ReLU')
        outputs = ZeroPadding2D(self.CONV_PADDING)(dec0)
        outputs = Conv2D(output_channel_count,
                         self.CONV_FILTER_SIZE,
                         strides=self.CONV_STRIDE)(outputs)
        outputs = Activation(activation='sigmoid')(outputs)

        self.UNET = Model(input=inputs, output=outputs)

    def _add_encoding_layer(self, filter_count, sequence):

        # max pooling
        new_sequence = ZeroPadding2D(self.MAX_POOLING_PADDING)(sequence)
        new_sequence = MaxPooling2D(pool_size=self.MAX_POOLING_SIZE,
                                    strides=self.MAX_POOLING_STRIDE)(new_sequence)

        # CONV BLOCK : convolution + activation function + batch norm
        new_sequence = self._add_convolution_block(filter_count, new_sequence, 'ReLU')
        new_sequence = self._add_convolution_block(filter_count, new_sequence, 'ReLU')

        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):

        new_sequence = self._add_convolution_block(filter_count*2, sequence, 'ReLU')
        #new_sequence = self._add_convolution_block(filter_count*2, new_sequence, 'ReLU')

        # up-convolution
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE,
                                       strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)

        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def _add_convolution_block(self, filter_count, sequence, act_function):

        #  CONV BLOCK : convolution + activation function + batch norm
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE,
                              strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if act_function == 'LeakyReLU':
            new_sequence = LeakyReLU(0.2)(new_sequence)
        elif act_function == 'ReLU':
            new_sequence = Activation(activation='relu')(new_sequence)

        return new_sequence

    def get_model(self):
        return self.UNET