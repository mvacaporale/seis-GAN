#------------------------------
# Custom Layers - For Verbosity
#------------------------------

import tensorflow as tf
import numpy      as np

class Transpose(tf.keras.layers.Layer):
    def __init__(self, perm, verbose = False, **kwargs):
        super(Transpose, self).__init__(**kwargs)
        self.verbose = verbose
        self.perm    = perm

    def __repr__(self):
        return self.name + ': perm = {}'.format(self.perm)

    def call(self, x):
        x = tf.transpose(x, perm = self.perm)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Split(tf.keras.layers.Layer):
    def __init__(self, axis, num_or_size_splits, verbose = False, **kwargs):
        super(Split, self).__init__(**kwargs)
        self.verbose            = verbose
        self.axis               = axis
        self.num_or_size_splits = num_or_size_splits

    def __repr__(self):
        return self.name + ': split = {}'.format(self.perm)

    def call(self, x):
        x = tf.split(x, num_or_size_splits = self.num_or_size_splits, axis = self.axis)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class ExpandDim(tf.keras.layers.Layer):
    def __init__(self, axis, verbose = False, **kwargs):
        super(ExpandDim, self).__init__(**kwargs)
        self.verbose = verbose
        self.axis    = axis

    def __repr__(self):
        return self.name + ': axis = {}'.format(self.axis)

    def call(self, x):
        x = tf.expand_dims(x, self.axis)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Squeeze(tf.keras.layers.Layer):
    def __init__(self, axis, verbose = False, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.verbose = verbose
        self.axis    = axis

    def __repr__(self):
        return self.name + ': axis = {}'.format(self.axis)

    def call(self, x):
        x = tf.squeeze(x, self.axis)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Reshape(tf.keras.layers.Layer):
    def __init__(self, shape, verbose = False, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.verbose = verbose
        self.shape    = shape

    def __repr__(self):
        return self.name + ': shape = {}'.format(self.shape)

    def call(self, x):
        x = tf.reshape(x, self.shape)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class ResizeNearestNeighbor(tf.keras.layers.Layer):
    def __init__(self, scale, verbose = False, align_corners = True, data_format = 'channels_last', **kwargs):
        super(ResizeNearestNeighbor, self).__init__(**kwargs)
        self.verbose       = verbose
        self.scale         = scale # 2-tuple to scale the shape
        self.align_corners = align_corners
        self.data_format   = data_format

    def __repr__(self):
        return self.name + ': scale = {}'.format(self.scale)

    def call(self, x):
        '''
        Resizes with resize_nearest_neighbor. Assumes either NWHC or NCHW format.
        '''
        size = [x.shape.as_list()[1] * self.scale[0], x.shape.as_list()[2] * self.scale[1]] if self.data_format == 'channels_last' else [x.shape.as_list()[2] * self.scale[0], x.shape[3] * self.scale.as_list()[1]]
        size = tuple(np.floor(size).astype(int))
        x    = tf.image.resize_nearest_neighbor(x, size = size, align_corners = self.align_corners)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class AveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, ksize, strides, padding, verbose = False, data_format = 'channels_last', **kwargs):
        super(AveragePooling2D, self).__init__(**kwargs)
        self.verbose       = verbose
        self.ksize         = ksize
        self.strides       = strides
        self.padding       = padding.upper() # Case sensitive.
        self.data_format   = data_format

    def __repr__(self):
        return self.name + ': ksize = {}'.format(self.ksize) + ', strides = {}'.format(self.strides) + ', padding = {}'.format(self.padding)

    def call(self, x):
        x = tf.nn.avg_pool(x, ksize = self.ksize, strides = self.strides, padding = self.padding)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class AveragePooling1D(AveragePooling2D):
    def __init__(self, ksize, strides, *args, **kwargs):
        ksize   = list(ksize  [0:2]) + [1, ksize  [2]]
        strides = list(strides[0:2]) + [1, strides[2]]
        super(AveragePooling1D, self).__init__(ksize, strides, *args, **kwargs)
        self.expand  = ExpandDim(axis = 2, verbose = kwargs.get('verbose', False))
        self.squeeze = Squeeze  (axis = 2, verbose = kwargs.get('verbose', False))

    def __call__(self, x):
        x = self.expand(x)
        x = super(AveragePooling1D, self).__call__(x)
        x = self.squeeze(x)
        return x

class ReLU(tf.keras.layers.Layer):
    def __init__(self, verbose = False, **kwargs):
        super(ReLU, self).__init__(**kwargs)
        self.verbose = verbose

    def call(self, x):
        x = tf.nn.relu(x)
        if self.verbose: print(self.name)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class LeakyReLU(tf.keras.layers.Layer):
    def __init__(self, verbose = False, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.verbose = verbose

    def call(self, x):
        x = tf.nn.leaky_relu(x)
        if self.verbose: print(self.name)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Tanh(tf.keras.layers.Layer):
    def __init__(self, verbose = False, **kwargs):
        super(Tanh, self).__init__(**kwargs)
        self.verbose = verbose

    def call(self, x):
        x = tf.nn.tanh(x)
        if self.verbose: print(self.name)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class BatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, verbose = False, **kwargs):
        super(BatchNormalization, self).__init__(**kwargs)
        self.verbose = verbose  

    def __repr__(self):
        return self.name + ': ' + 'axis = ' + str(self.axis)

    def call(self, x, **kwargs):
        x = super(BatchNormalization, self).call(x, **kwargs)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Dense(tf.layers.Dense):
    def __init__(self, units, verbose = False, **kwargs):
        super(Dense, self).__init__(units = units, **kwargs)
        self.verbose = verbose 
        self.units   = units

    def __repr__(self):
        try:
            return self.name + ': ' + 'input [{}]'.format(self.input_shape) + 'output [{}]'.format(self.output_shape)
        except AttributeError:
            return self.name + ': ' + 'units = {}'.format(self.units)

    def call(self, x, **kwargs):
        x = super(Dense, self).call(x, **kwargs)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Conv2D(tf.layers.Conv2D):
    def __init__(self, filters, kernel_size, strides, verbose = False, **kwargs):
        super(Conv2D, self).__init__(filters = filters, kernel_size = kernel_size, strides = strides, **kwargs)
        self.verbose     = verbose 
        self.filters     = filters
        self.kernel_size = kernel_size
        self.strides     = strides

    def __repr__(self):
        return self.name + ': ' + 'filters = {} // '.format(self.filters) + 'kernel_size = {} // '.format(self.kernel_size) + 'strides = {} '.format(self.strides) 

    def call(self, x, **kwargs):
        x = super(Conv2D, self).call(x, **kwargs)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

class Flatten(tf.layers.Flatten):
    def __init__(self, verbose = False, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.verbose     = verbose 

    def __repr__(self):
        return self.name

    def call(self, x, **kwargs):
        x = super(Flatten, self).call(x, **kwargs)
        if self.verbose: print(self)
        if self.verbose: print(' '*5 + 'x.shape', [xi.shape for xi in x] if isinstance(x, list) else x.shape)
        return x

