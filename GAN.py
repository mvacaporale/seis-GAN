import tensorflow as tf
import numpy      as np
from   copy           import deepcopy
from   collections    import namedtuple
from   ProgressiveGAN import ProgressiveGAN
from   Layers         import *

#---------------------------
# Generator & Discriminator
#---------------------------
class Generator(tf.keras.Model, metaclass = ProgressiveGAN):

    #--------------------
    # Progressive Layers
    #--------------------

    # Define convolutional layer to adjust the filter size and other dimensions for the final output sample.
    class ToSample(Conv2D):
        def __init__(self, *args, **kwargs):
            super(Generator.ToSample, self).__init__(filters = 3, kernel_size = (1, 1), strides = (1, 1), padding = 'same', *args, **kwargs)
            self.tanh      = Tanh(verbose = kwargs.get('verbose', False))
            self.squeeze   = Squeeze(axis = 2, verbose = kwargs.get('verbose', False))
            self.transpose = Transpose(perm = [0, 1, 2] if self.data_format == 'channels_last' else [0, 2, 1], verbose = kwargs.get('verbose', False))

        def call(self, x, **kwargs):
            x = super(Generator.ToSample, self).call(x, **kwargs)
            x = self.tanh(x)
            x = self.squeeze(x)
            x = self.transpose(x)
            return x

    # Define layer to resize generated output of early stages. 
    class Resize(ResizeNearestNeighbor):
        def __init__(self, scale, data_format = 'channels_last', *args, **kwargs):
            super(Generator.Resize, self).__init__(*args, (scale, 1), data_format = data_format, *args, **kwargs)
            self.temp_index = 2 if data_format == 'channels_last' else 3
            self.expand     = ExpandDim(axis = self.temp_index, verbose = kwargs.get('verbose', False))
            self.squeeze    = Squeeze  (axis = self.temp_index, verbose = kwargs.get('verbose', False))

        def call(self, x):
            x = self.expand(x)
            x = super(Generator.Resize, self).call(x)
            x = self.squeeze(x)
            return x

    # Define block in progression.
    class Block(tf.keras.layers.Layer):
        def __init__(self, filters, kernel_size, strides, padding, data_format = 'channels_last', verbose = False, *args, **kwargs):
            super(Generator.Block, self).__init__(*args, **kwargs)

            self.resize      = ResizeNearestNeighbor(scale = (2, 1), verbose = verbose)
            
            self.conv1       = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, verbose = verbose)
            self.batchnorm1  = BatchNormalization(axis = -1, verbose = verbose)
            self.relu1       = ReLU(verbose = verbose)
            
            self.conv2       = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, data_format = data_format, verbose = verbose)
            self.batchnorm2  = BatchNormalization(axis = -1, verbose = verbose)
            self.relu2       = ReLU(verbose = verbose)

        def call(self, x, training = True):

            x = self.resize(x)

            x = self.conv1(x)
            x = self.batchnorm1(x, training = training)
            x = self.relu1(x)

            x = self.conv2(x) 
            x = self.batchnorm2(x, training = training) 
            x = self.relu2(x)
            return x

    #--------------------
    # Output Utilities
    #--------------------

    # Define named tuple to return output from call to generator.
    gen_output = namedtuple('gen_output', [
        'data'     , # generated data
        'pred_norm', # predicted normalization factor of generated data (if predict_normalization_factor = True on call)
        ] 
    )
    gen_output.__new__.__defaults__ = (None,) * len(gen_output._fields) # Set defaults to None

    def __init__(self, data_format = 'channels_first', verbose = False, predict_normalization_factor = False):

        super(Generator, self).__init__()

        # Init params.
        self.data_format  = data_format
        self.verbose      = verbose
        self.predict_norm = predict_normalization_factor

        # Init Layers
        self.c0 = Dense(units = 200, use_bias = False, verbose = verbose)
        self.transpose   = Transpose(perm = [0, 2, 1] if self.data_format == 'channels_last' else [0, 1, 2], verbose = verbose)

        self.batchnorm00 = BatchNormalization(axis = 1, verbose = verbose)
        self.fc00        = Dense(units = 200, use_bias = False, verbose = verbose)

        self.transpose00 = Transpose(perm = [0, 2, 1], verbose = verbose)
        self.expand00    = ExpandDim(axis = 2, verbose = verbose)

        self.conv0       = Conv2D(3, kernel_size = (16, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)
        self.batchnorm0  = BatchNormalization(axis = -1, verbose = verbose)
        self.relu0       = ReLU(verbose = verbose)

        self.conv0b      = Conv2D(6, kernel_size = (16, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)
        self.batchnorm0b = BatchNormalization(axis = -1, verbose = verbose)
        self.relu0b      = ReLU(verbose = verbose)

        self.conv0c      = Conv2D(3, kernel_size = (16, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)
        self.batchnorm0c = BatchNormalization(axis = -1, verbose = verbose)
        self.relu0c      = ReLU(verbose = verbose)

        self.squeeze0c   = Squeeze(axis = 2, verbose = verbose)
        self.transpose0c = Transpose(perm = [0, 2, 1], verbose = verbose)

        self.fc01_n      = 400 + (1 if self.predict_norm else 0)
        self.fc01        = Dense(units = self.fc01_n, use_bias = False, verbose = verbose)
        self.batchnorm01 = BatchNormalization(axis = 1, verbose = verbose)
        self.relu01      = ReLU(verbose = verbose)

        self.reshape01   = Reshape(shape = (-1, 100, 1, 12), verbose = verbose)

        self.block1    = self.Block(32, kernel_size = (4, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)
        self.block2    = self.Block(64, kernel_size = (4, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)
        self.block3    = self.Block(32, kernel_size = (4, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)
        self.block4    = self.Block(16, kernel_size = (4, 1), strides = (1, 1), padding = 'same', data_format = self.data_format, verbose = verbose)

        self.resize1   = self.Resize(scale = 2, verbose = verbose)
        self.resize2   = self.Resize(scale = 2, verbose = verbose)
        self.resize3   = self.Resize(scale = 2, verbose = verbose)
        self.resize4   = self.Resize(scale = 2, verbose = verbose)

        self.to_samp0 = self.ToSample(data_format = self.data_format, verbose = self.verbose)
        self.to_samp1 = self.ToSample(data_format = self.data_format, verbose = self.verbose)
        self.to_samp2 = self.ToSample(data_format = self.data_format, verbose = self.verbose)
        self.to_samp3 = self.ToSample(data_format = self.data_format, verbose = self.verbose)
        self.to_samp4 = self.ToSample(data_format = self.data_format, verbose = self.verbose)

        self.blocks     = [self.block1, self.block2, self.block3, self.block4]
        self.resizings  = [self.resize1, self.resize2, self.resize3, self.resize4]
        self.to_samps   = [self.to_samp0, self.to_samp1, self.to_samp2, self.to_samp3, self.to_samp4]

    def call(self, x, conditional = None, training = True):

        if self.verbose: print('Generator.call()', 'training = {}'.format(training))
        if self.verbose: print(' '*5 + 'x.shape', x.shape if not isinstance(x, list) else None)
        if self.verbose: print()
        # self.c0.build(x.shape)

        y = self.c0(x)

        if conditional is not None:

            if self.verbose: print('Appending Conditional:')
            conditional = tf.concat(conditional, axis = -1) if isinstance(conditional, list) else conditional
            if self.verbose: print(' '*5 + 'conditional:', conditional, conditional.shape)
            conditional = tf.tile(conditional[:, :, None], [1, 1, 3])
            if self.verbose: print(' '*5 + 'conditional:', conditional, conditional.shape)
            # conditional = conditional.transpose((1, 0, 2)) if self.data_format == 'channel_first' else conditional.transpose((1, 2, 0))
            x = tf.concat([x, conditional], 2 if self.data_format == 'channels_first' else 1)

            if self.verbose: print(' '*5 + 'added_conditional')
            if self.verbose: print(' '*5 + 'x.shape', x.shape if not isinstance(x, list) else None)
            if self.verbose: print()

        x = self.transpose(x)

        x = self.fc00(x) 
        x = self.batchnorm00(x, training = training)

        x = self.transpose00(x)
        x = self.expand00(x)

        x = self.conv0(x)
        x = self.batchnorm0(x, training = training)    
        x = self.relu0(x)

        x = self.conv0b(x)
        x = self.batchnorm0b(x, training = training)
        x = self.relu0b(x)

        x = self.conv0c(x)
        x = self.batchnorm0c(x, training = training)
        x = self.relu0c(x)

        x = self.squeeze0c(x)
        x = self.transpose0c(x)

        x = self.fc01(x)  

        if self.predict_norm:
            x, pred_norm = tf.split(x, [self.fc01_n - 1, 1], axis = -1) 
            pred_norm    = pred_norm[..., -1]
        else:
            pred_norm = None

        x = self.batchnorm01(x, training = training)
        x = self.relu01(x)

        x = self.reshape01(x)

        return self.gen_output(x, pred_norm)

class Discriminator(tf.keras.Model, metaclass = ProgressiveGAN):

    #--------------------
    # Progressive Layers
    #--------------------

    # Define layer from convolutional output shape to sample shape.
    class FromSample(Conv2D):
        def __init__(self, filters, verbose = False, *args, **kwargs):
            super(Discriminator.FromSample, self).__init__(filters = filters, kernel_size = (1, 1), strides = (1, 1), padding = 'same', verbose = verbose, *args, **kwargs)
            self.expand  = ExpandDim(axis = 2, verbose = verbose)

        def __call__(self, x, **kwargs):
            x = self.expand(x) # Expand here to avoid any issue with dim checks in __call__
            x = super(Discriminator.FromSample, self).__call__(x, **kwargs)
            return x

    class Block(tf.keras.layers.Layer):
        def __init__(self, filters, data_format = 'channels_last', verbose = False, **kwargs):
            super(Discriminator.Block, self).__init__(**kwargs)
            self.verbose = verbose

            self.conv1   = Conv2D(filters, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = data_format, verbose = verbose)
            self.leaky1 = LeakyReLU(verbose = verbose)

            self.conv2   = Conv2D(filters, kernel_size = (4, 1), strides = (1, 1), padding = 'same', data_format = data_format, verbose = verbose)
            self.leaky2  = LeakyReLU(verbose = verbose)

        def call(self, x, training = True):

            x = self.conv1(x)
            x = self.leaky1(x)

            x = self.conv2(x)
            x = self.leaky2(x)

            return x

    #--------------------
    # Output Utilities
    #--------------------

    # Define named tuple to return output from call to discriminator.
    dis_output = namedtuple('dis_output', [
        'score'      , # score of input data (the measure of "realness")
        'extra_loss' , # loss terms to add to training scheme (used when aux_classify = True)
        ] 
    )

    #--------------------
    # Class Methods
    #--------------------

    def __init__(self, data_format = 'channels_first', verbose = False, aux_classify = False, aux_categories_sizes = []):
        super(Discriminator, self).__init__()

        self.aux_classify = aux_classify
        self.data_format  = data_format
        self.verbose      = verbose

        if self.aux_classify: 
            self.ac_dims = aux_categories_sizes
            self.ac_n    = 50
        else:
            self.ac_dims = []
            self.ac_n = 0

        self.c = 32 # hyperparam - size of filter for last conv.
        self.block1 = self.Block(filters = 16    , verbose = False)
        self.block2 = self.Block(filters = 32    , verbose = False)
        self.block3 = self.Block(filters = 32    , verbose = False)
        self.block4 = self.Block(filters = self.c, verbose = False)

        self.resize1 = AveragePooling1D(ksize = (1, 2, 1, 1), strides = (1, 2, 1, 1), padding = 'same', verbose = False, data_format = self.data_format)
        self.resize2 = AveragePooling1D(ksize = (1, 2, 1, 1), strides = (1, 2, 1, 1), padding = 'same', verbose = False, data_format = self.data_format)
        self.resize3 = AveragePooling1D(ksize = (1, 2, 1, 1), strides = (1, 2, 1, 1), padding = 'same', verbose = False, data_format = self.data_format)
        self.resize4 = AveragePooling1D(ksize = (1, 2, 1, 1), strides = (1, 2, 1, 1), padding = 'same', verbose = False, data_format = self.data_format)

        self.from_samp0 = self.FromSample(filters = 3     , verbose = False)
        self.from_samp1 = self.FromSample(filters = 16    , verbose = False)
        self.from_samp2 = self.FromSample(filters = 32    , verbose = False)
        self.from_samp3 = self.FromSample(filters = 32    , verbose = False)
        self.from_samp4 = self.FromSample(filters = self.c, verbose = False)

        self.blocks     = [self.block1, self.block2, self.block3, self.block4]
        self.resizings  = [self.resize1, self.resize2, self.resize3, self.resize4]
        self.from_samps = [self.from_samp0, self.from_samp1, self.from_samp2, self.from_samp3, self.from_samp4]

        self.squeeze     = Squeeze(axis = 2, verbose = verbose)
        self.transpose2b = Transpose(perm = [0, 2, 1])

        self.fc0_n      = 300 # 110 # hyperparam - units of dense layer without any dedicated to aux-classification
        self.fc0        = Dense(self.fc0_n + self.ac_n * len(self.ac_dims), verbose = verbose)
        self.leakyf0    = LeakyReLU(verbose = verbose)

        self.fc1        = Dense(150, verbose = verbose)
        self.leakyf1    = LeakyReLU(verbose = verbose)

        self.fc1b       = Dense(150, verbose = verbose)
        self.leakyf1b   = LeakyReLU(verbose = verbose)

        self.flatten    = Flatten(verbose = verbose)
        self.fc2        = Dense(1, verbose = verbose)

    def call(self, x, conditional = None, pred_norm = None, training = True):
        
        if self.verbose: print('Discriminator.call()', training)
        if self.verbose: print(' '*5 + 'x.shape', x.shape if not isinstance(x, list) else None)

        x = self.squeeze(x)
        
        if conditional is not None and not self.aux_classify:

            if self.verbose: print('Appending Conditional:')
            conditional = tf.concat(conditional, axis = -1) if isinstance(conditional, list) else conditional
            if self.verbose: print(' '*5 + 'conditional:', conditional, conditional.shape)
            conditional = tf.tile(conditional[:, :, None], [1, 1, self.c])
            if self.verbose: print(' '*5 + 'conditional:', conditional, conditional.shape)
            # conditional = conditional.transpose((1, 0, 2)) if self.data_format == 'channel_first' else conditional.transpose((1, 2, 0))
            x = tf.concat([x, conditional], 2 if self.data_format == 'channels_first' else 1)

            if self.verbose: print(' '*5 + 'appended conditional')
            if self.verbose: print(' '*5 + 'x.shape', x.shape if not isinstance(x, list) else None)

        if pred_norm is not None:

            if self.verbose: print('Appending Predicted Norm:')
            if self.verbose: print(' '*5 + 'pred_norm:', pred_norm, pred_norm.shape)
            pred_norm = tf.tile(pred_norm[:, :, None], [1, 1, self.c])
            if self.verbose: print(' '*5 + 'pred_norm:', pred_norm, pred_norm.shape)
            x = tf.concat([x, pred_norm], 2 if self.data_format == 'channels_first' else 1)

            if self.verbose: print(' '*5 + 'appended predicted norm')
            if self.verbose: print(' '*5 + 'x.shape', x.shape if not isinstance(x, list) else None)

        x = self.transpose2b(x)

        x = self.fc0(x)
        x = self.leakyf0(x)

        loss = tf.constant(0, tf.float32)
        if self.aux_classify and conditional is not None:

            conditional = [conditional] if not isinstance(conditional, list) else conditional
            x           = tf.split(x, [self.fc0_n] + [self.ac_n] * len(conditional), axis = -1)

            x_acs = x[1:]
            x     = x[0] 

            for i, cond in enumerate(conditional):
                c_dim = cond.shape[-1]
                x_ac  = tf.expand_dims(x_acs[i], -1, name = 'aux_input')
                x_ac  = tf.transpose(x_ac, perm = [0, 2, 3, 1])
                x_ac  = self.ac_layers(x_ac, c_dim)
                loss  +=  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = x_ac, labels = conditional[i]))

        x = self.fc1(x)
        x = self.leakyf1(x)

        x = self.fc1b(x)
        x = self.leakyf1b(x)

        x = self.flatten(x)
        x = self.fc2(x)
        x = tf.identity(x, 'output')
        
        return self.dis_output(x, loss)

class GAN(object):
    
    def __init__(self, noise_dim = 100, data_format = 'channels_first', verbose = False, aux_classify = False, aux_categories_sizes = [],  predict_normalization_factor = False):
            
        self.data_format   = data_format
        self.noise_dim     = noise_dim
        self.predict_norm  = predict_normalization_factor
        self.generator     = Generator    (data_format = data_format, verbose = verbose, predict_normalization_factor = self.predict_norm)
        self.discriminator = Discriminator(data_format = data_format, verbose = verbose, aux_classify = aux_classify, aux_categories_sizes = aux_categories_sizes)

#------------------------------------------------------
# Running a test forward pass through both networks.
#------------------------------------------------------
if __name__ == '__main__':
    
    import numpy as np

    # Init gan and gen / dis networks.
    gan           = GAN(data_format = 'channels_last', verbose = False, predict_normalization_factor = True) # norm predicting gan
    generator     = gan.generator
    discriminator = gan.discriminator

    # Defun gives 10 secs/epoch performance boost
    # generator.call     = tf.contrib.eager.defun(generator.call)
    # discriminator.call = tf.contrib.eager.defun(discriminator.call)

    # Create random initial vector.
    num_examples_to_generate     = 4
    random_vector_for_generation = tf.random_normal([num_examples_to_generate, 3, gan.noise_dim]) if gan.data_format == 'channels_first' else tf.random_normal([num_examples_to_generate, gan.noise_dim, 3]) 

    # Create conditional input if desired.
    if True:
        conditional = np.random.normal(size = (num_examples_to_generate, 3)).astype(np.float32)
    else:
        conditional = None

    #------------------------------------------------------
    # Run a test forward pass through both networks.
    #------------------------------------------------------

    # Generate sample data.
    gen_output = generator(random_vector_for_generation, conditional = [conditional])  
    gen        = gen_output.data    
    pred_norm  = gen_output.pred_norm   

    # Discriminate mimic / random real data. This allows the Dis network to initialize.
    real_input = tf.identity(tf.random_normal([4, 1600, 3]), 'real_input')
    dis        = discriminator(real_input, training = False, conditional = [conditional], pred_norm = pred_norm).score

    # Prep real input - i.e. resize input if this is a ProgressiveGAN
    if hasattr(discriminator, 'prep_real_input'):
        real_input = discriminator.prep_real_input(real_input)

    # Run real_input through dis as well as generated sample data.
    dis        = discriminator(real_input, conditional = [conditional], pred_norm = pred_norm).score
    dis        = discriminator(gen       , conditional = [conditional], pred_norm = pred_norm).score

    # Print out model summary.
    with open('model_summary.txt', 'w') as f:
        f.write('Generator Summary:\n')
        s = generator.summary(print_fn = lambda summary: [f.write(summary + '\n'), print(summary) if generator.verbose else None])
        f.write('\n\n\n\n\n')

        f.write('Discriminator Summary:\n')
        s = discriminator.summary(print_fn = lambda summary: [f.write(summary + '\n'), print(summary) if discriminator.verbose else None])
        f.write('\n')
