# Based on mine, NVIDIA implemtation for TF1 https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet
# and Miguel Monteiro https://github.com/MiguelMonteiro/VNet-Tensorflow
## Author: {Pedro M. Gordaliza}
## Copyright: Copyright {year}, {project_name}
## Credits: [{NVIDIA}]
## License: {license}
## Version: {0}.{1}.{0}
## Mmaintainer: {maintainer}
## Email: {pedro.macias.gordaliza@gmail.com}
## Status: {dev}

import tensorflow as tf



ACTIVATIONS_DICT = {
    None:None,
    "relu":tf.nn.relu
}
NORMALIZATION_LAYERS_LIST = [None, 'batchnorm']
UPAMPLING_LIST = ['transposed_conv']


def residual_block(input_0, input_1, kernel_size, depth, training=True, **kwargs):
    """
    kwargs for ConvolutionalLayer
    """
    with tf.name_scope('residual_block'):
        x = input_0
        if input_1 is not None:
            x = tf.concat([input_0, input_1], axis=-1)

        inputs = x
        n_input_channels = inputs.get_shape()[-1]

        for i in range(depth):
            cvn = ConvolutionalLayer(filters=n_input_channels, kernel_size=kernel_size, stride=1, **kwargs)
            x = cvn(x, training=training)

        return x + inputs

class ResidualBlock(tf.keras.Model):
    def __init__(self, depth, **kwargs):
        super().__init__(self)
        with tf.name_scope('residual_block'):
            self._depth = depth
            self.convolutions_block = [ConvolutionalLayer(stride=1, **kwargs) for cvn in range(self._depth)]
    
    def call(self, input_0, input_1=None, training=True):
        x = input_0
        if input_1 is not None:
            x = tf.concat([input_0, input_1], axis=-1)
        inputs = x
        
        for cvn in self.convolutions_block:
            x = cvn(x, training=training)
        return x + inputs
            
        
def downsample_layer(inputs, pooling, training = True, **kwargs):
    """
    kwargs for Convolutional layer, except for filters set to input channels *2
    kernel size = 2 and stride = 2
    """
    with tf.name_scope('downsample_layer'):
        if pooling == 'conv_pool':
            cvn = ConvolutionalLayer(inputs.get_shape()[-1] * 2, kernel_size=2, stride = 2, **kwargs)
            return cvn(inputs, training=training)
        else:
            raise ValueError('Invalid downsampling method: {}'.format(pooling))
            
class DownsampleLayer(tf.keras.Model):
    def __init__(self, pooling, **kwargs):
        super().__init__(self)
        with tf.name_scope('downsample_layer'):
            self._pooling = pooling #TODO properly with set/get and choices as normalizationlayer and so on
            if self._pooling == 'conv_pool':
                self.cvn = ConvolutionalLayer(kernel_size=2, stride = 2, **kwargs)
            else:
                raise ValueError('Invalid downsampling method: {}'.format(pooling))
            
    def call(self, features, training=True):
        return self.cvn(features, training=training)
        

class Normalization_Layer(tf.keras.Model):
    BN = 'batchnorm'
    def __init__(self, name):
        super().__init__(self)
        self.norm_type = name
        self.norm_layer = None
        if name == Normalization_Layer.BN:
            self.norm_layer = tf.keras.layers.BatchNormalization()
        
    @property
    def norm_type(self):
        return self.__norm_type
    @norm_type.setter
    def norm_type(self, name):
        if name in NORMALIZATION_LAYERS_LIST:
            self.__norm_type = name
        else:
            raise ValueError('Invalid normalization layer '+name)
            
    def call(self, inputs, training = True):
        return inputs if self.norm_layer is None else self.norm_layer(inputs, training=training)
    
class ConvolutionalLayer(tf.keras.Model):
    def __init__(self, filters, kernel_size, stride, normalization=Normalization_Layer.BN, conv_activation=None, final_activation="relu"):
        super().__init__(self)
        with tf.name_scope('convolution_layer'):
            self.convolution = tf.keras.layers.Conv3D(filters, kernel_size, stride, activation=conv_activation, padding='same')
            self.norm_layer = Normalization_Layer(normalization)
            self.final_activation = final_activation
    @property
    def final_activation(self):
        return self.__final_activation
    @final_activation.setter
    def final_activation(self, activation):
        if activation in ACTIVATIONS_DICT.keys():
            self.__final_activation = ACTIVATIONS_DICT[activation]
        else: 
            raise ValueError("Unkown activation {}".format(activation))
        
    def call(self, features, training=True):
        out = self.convolution(features)
        out = self.norm_layer(out, training=training)
        return out if self.final_activation is None else self.final_activation(out)
        

class UpsampleLayer(tf.keras.Model):
    #TODO UPSAMPLING POOLING and ACTIVATIONS layers to generalize in the correspodinf classes/methods
    def __init__(self, filters, upsampling='transposed_conv', kernel_size = 2, strides = 2, conv_activation=None, final_activation='relu', normalization=Normalization_Layer.BN):
        super().__init__(self)
        with tf.name_scope('upsample_layer'):
            self.upsampling = upsampling
            self.trans_convolution = tf.keras.layers.Conv3DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same', activation=conv_activation)
            self.norm_layer = Normalization_Layer(normalization)
            self.final_activation = final_activation
    @property
    def final_activation(self):
        return self.__final_activation
    @final_activation.setter
    def final_activation(self, activation):
        if activation in ACTIVATIONS_DICT.keys():
            self.__final_activation = ACTIVATIONS_DICT[activation]
        else: 
            raise ValueError("Unkown activation {}".format(activation))
            
    @property
    def upsampling(self):
        return self.__upsampling
    @upsampling.setter
    def upsampling(self, upsampling):
        if upsampling in UPAMPLING_LIST:
            self.__upsampling = upsampling
        else:
            raise ValueError('Unsupported upsampling: {}'.format(upsampling))
     
    def call(self, features, training = True):
        out = self.trans_convolution(features)
        out = self.norm_layer(out, training=training)
        return out if self.final_activation is None else self.final_activation(out)
    
class InputBlock(tf.keras.Model):
    def __init__(self, **kwargs):
        """
        kwargs for ConvolutionaLayer, except for stride which is fix to 1
        """
        super().__init__(self)
        with tf.name_scope('input_block'):
            self.conv = ConvolutionalLayer(stride = 1, **kwargs)
            
    def call(self, inputs, training=True):
        x = inputs
        # residual is included here directly (e.g. +x)
        return self.conv(inputs, training=training) + x


class DownsampleBlock(tf.keras.Model):
    def __init__(self, res_depth, res_kernel_size, res_filters, down_pooling, down_filters, normalization=Normalization_Layer.BN, conv_activation=None, final_activation='relu'):
        super().__init__(self)
        with tf.name_scope('downsample_block'):
            self._depth = res_depth
            self._pooling = down_pooling
            self._down_filters = down_filters 
            self._kernel_size = res_kernel_size
            self._res_filters = res_filters
            self._normalization = normalization
            self._conv_act = conv_activation
            self._final_act = final_activation
    
            self.down_layer = DownsampleLayer(filters=self._down_filters, pooling=self._pooling, normalization=self._normalization, conv_activation=self._conv_act, final_activation=self._final_act)
            self.residual_block = ResidualBlock(depth=self._depth, filters=self._res_filters, kernel_size=self._kernel_size, normalization=self._normalization, conv_activation=self._conv_act, final_activation=self._final_act)
        
    def call(self, features, training = True):
        #x = downsample_layer(features, pooling=self._pooling, normalization=self._normalization, conv_activation=self._conv_act, final_activation=self._final_act, training=training)

        x = self.down_layer(features, training=training)
        return self.residual_block(x, input_1=None, training=training)
        #return residual_block(x, None, depth=self._depth, kernel_size=self._kernel_size, normalization=self._normalization, conv_activation=self._conv_act, final_activation=self._final_act, training=training)


class UpsampleBlock(tf.keras.Model):
    def __init__(self, res_depth, res_kernel_size, res_filters, up_filters, upsampling='transposed_conv', normalization=Normalization_Layer.BN, conv_activation=None, final_activation='relu'):
        super().__init__(self)
        with tf.name_scope('upsample_block'):
            self._res_depth = res_depth
            self._res_kernel_size = res_kernel_size
            self._res_filters = res_filters
            self._up_filters = up_filters
            self._upsampling = upsampling
            self._normalization = normalization
            self._conv_act = conv_activation
            self._final_act = final_activation
            
            self.upsample_layer = UpsampleLayer(filters=up_filters, upsampling=self._upsampling, normalization=self._normalization, conv_activation=self._conv_act, final_activation=self._final_act)
            self.residual_block = ResidualBlock(depth=self._res_depth, filters=self._res_filters, kernel_size=self._res_kernel_size,normalization=self._normalization,conv_activation=self._conv_act, final_activation=self._final_act)
            
    def call(self, features, residual_inputs, training = True):
        #up_cvn = UpsampleLayer(filters=residual_inputs.get_shape()[-1],upsampling=self._upsampling,normalization=self._normalization,conv_activation=self._conv_act,final_activation=self._final_act)
        x = self.upsample_layer(features, training=training)
        #x = up_cvn(features, training=training)
        return self.residual_block(x, input_1=residual_inputs, training=training)

    
class OutputBlock(tf.keras.Model):
    def __init__(self, n_classes, con_kernel_size, up_filters, upsampling='transposed_conv', up_normalization=Normalization_Layer.BN, up_conv_activation=None, up_final_activation='relu'):
        super().__init__(self)
        with tf.name_scope('output_block'):
            self._up_filters = up_filters
            self._upsampling = upsampling
            self._up_normalization = up_normalization
            self._up_conv_act = up_conv_activation
            self._up_final_act = up_final_activation
            self.cvn = ConvolutionalLayer(filters=n_classes, kernel_size=con_kernel_size, stride=1, conv_activation=None, final_activation=None, normalization=None)
            self.upsample_layer = UpsampleLayer(filters=self._up_filters, upsampling=self._upsampling, normalization=self._up_normalization, conv_activation=self._up_conv_act, final_activation=self._up_final_act) 
        
    def call(self, features, residual_inputs, training = True):
        #up_cvn = UpsampleLayer(filters=residual_inputs.get_shape()[-1], upsampling=self._upsampling, normalization=self._up_normalization, conv_activation=self._up_conv_act, final_activation=self._up_final_act)
        x = self.upsample_layer(features, training=training)
        return self.cvn(x, training=training) # the original made a 1x1x1 covolution to do this tanformation with a previous residual convolution
    

    

if __name__ == '__main__':
    import numpy as np
    cvn = ConvolutionalLayer(16, 3, 1, normalization=None)
    a = np.random.randint(10, size = (1,7,7,7,1)).astype(np.float32)
    res = cvn(a)
    res2 = cvn(a, training=False)
