import tensorflow as tf
import numpy      as np
from copy import deepcopy

class Generator(tf.keras.Model):
  def __init__(self, data_format = 'channels_first', verbose = False):
    super(Generator, self).__init__()
    
    self.data_format = data_format
    self.verbose     = verbose
    
    # Hyperparameters.
    # self.fcl_n = 200

    # Define layers.
    self.flatten     = tf.layers.Flatten()
    self.fc0         = tf.layers.Dense(120, use_bias = False)
    self.batchnorm00 = tf.layers.BatchNormalization(axis = 1)
    self.dropout0    = tf.layers.Dropout(0.5)

    self.fc1         = tf.layers.Dense(200, use_bias = False)
    self.batchnorm01 = tf.layers.BatchNormalization(axis = 1) # axis = 2 if self.data_format == 'channels_last' else 1
    self.dropout1    = tf.layers.Dropout(0.5)
    
    # self.conv1      = tf.layers.UpSampling1D(size = 2)
    self.conv1       = tf.layers.Conv2DTranspose(12, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = self.data_format)
    self.batchnorm1  = tf.layers.BatchNormalization(axis = 3 if self.data_format == 'channels_last' else 1)
    self.dropout1    = tf.layers.Dropout(0.5)
    
    # self.conv2      = tf.layers.Conv2DTranspose(6, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = self.data_format)
    # self.batchnorm2 = tf.layers.BatchNormalization(axis = 3 if self.data_format == 'channels_last' else 1)
    # self.dropout2   = tf.layers.Dropout(0.5)
    
    self.conv3       = tf.layers.Conv2DTranspose(3, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = self.data_format)

  def call(self, x, conditional = None, training = True):
    
    if self.verbose: print('Generator.call()', training)
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    if conditional is not None:

        if self.verbose: print('conditional:', conditional, conditional.shape)
        conditional = tf.tile(conditional[:, :, None], [1, 1, 3])
        if self.verbose: print('conditional:', conditional, conditional.shape)
        # conditional = conditional.transpose((1, 0, 2)) if self.data_format == 'channel_first' else conditional.transpose((1, 2, 0))
        x = tf.concat([x, conditional], 2 if self.data_format == 'channels_first' else 1)

        if self.verbose: print('added_conditional')
        if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.transpose(x, perm = [0, 2, 1]) if self.data_format == 'channels_last' else x
    if self.verbose: print('reshape')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.fc0(x)
    if self.verbose: print('fc1')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.batchnorm00(x, training = training)
    if self.verbose: print('batchnorm')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.dropout0(x, training = training)
    if self.verbose: print('droupout')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.fc1(x)
    if self.verbose: print('fc1')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.batchnorm01(x, training = training)
    if self.verbose: print('batchnorm')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.dropout1(x, training = training)
    if self.verbose: print('droupout')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.relu(x)
    if self.verbose: print('relu')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.reshape(x, shape=(-1, 100, 1, 6))
    if self.verbose: print('reshape')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    # x_s = [x]
    
    x = self.conv1(x)
    # x_s.append(deepcopy(x))
    if self.verbose: print('conv1')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.batchnorm1(x, training = training)
    # x_s.append(deepcopy(x))
    if self.verbose: print('batchnorm')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.dropout1(x, training=training)
    # x_s.append(deepcopy(x))
    if self.verbose: print('droupout')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.relu(x)
    # x_s.append(deepcopy(x))
    if self.verbose: print('relu')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    # x = self.conv2(x)
    # # x_s.append(deepcopy(x))
    # if self.verbose: print('conv1')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    # x = self.batchnorm2(x, training = training)
    # # x_s.append(deepcopy(x))
    # if self.verbose: print('batchnorm')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    # x = self.dropout2(x, training=training)
    # # x_s.append(deepcopy(x))
    # if self.verbose: print('droupout')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    # x = tf.nn.tanh(x)
    # # x_s.append(deepcopy(x))
    # if self.verbose: print('tanh')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.conv3(x)
    # x_s.append(deepcopy(x))
    if self.verbose: print('conv')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.tanh(x)
    # x_s.append(deepcopy(x))
    if self.verbose: print('tanh')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    # x = tf.nn.tanh( self.conv3(x) ) 
    if self.verbose: print('tanh')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.transpose(x, perm = [0, 2, 1]) if self.data_format == 'channels_first' else x
    # x_s.append(deepcopy(x))
    if self.verbose: print('reshape')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    return x

class Discriminator(tf.keras.Model):
  def __init__(self, data_format = 'channels_first', verbose = False):
    super(Discriminator, self).__init__()
    
    self.data_format = data_format
    self.verbose     = verbose
    
    # Define layers. 
    self.conv1      = tf.layers.Conv2D(16, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = self.data_format)
    self.conv2      = tf.layers.Conv2D(16, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = self.data_format)
    self.conv3      = tf.layers.Conv2D(16, kernel_size = (4, 1), strides = (2, 1), padding = 'same', data_format = self.data_format) 
    self.avgpool1   = tf.layers.AveragePooling2D( pool_size = (3, 1), strides = 2, padding = 'same', data_format = self.data_format)
    self.avgpool2   = tf.layers.AveragePooling2D( pool_size = (3, 1), strides = 2, padding = 'same', data_format = self.data_format)
    # self.dropout1   = tf.layers.Dropout(0.5)
    # self.dropout2   = tf.layers.Dropout(0.5)
    # self.batchnorm1 = tf.layers.BatchNormalization(axis = 2 if self.data_format == 'channels_last' else 1)
    # self.batchnorm2 = tf.layers.BatchNormalization(axis = 2 if self.data_format == 'channels_last' else 1)
    # self.batchnorm2 = tf.layers.BatchNormalization(axis = 2 if self.data_format == 'channels_last' else 1)
    self.flatten    = tf.layers.Flatten()
    self.fc0        = tf.layers.Dense(70)
    self.fc1        = tf.layers.Dense(30)
    self.fc2        = tf.layers.Dense(1)

  def call(self, x, conditional = None, training = True):
    
    if self.verbose: print('Discriminator.call()', training)
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.conv1(x)
    if self.verbose: print('conv')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.avgpool1(x)
    if self.verbose: print('average_pool')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.nn.leaky_relu(x)
    if self.verbose: print('leaky_relu')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
 
    x = self.conv2(x)
    if self.verbose: print('conv')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.avgpool2(x)
    if self.verbose: print('average_pool')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
   
    x = tf.nn.leaky_relu(x)
    if self.verbose: print('leaky_relu(conv)')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    # x = self.batchnorm1(x, training=training)
    # if self.verbose: print('batchnorm')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    # x = tf.nn.leaky_relu(self.conv3(x))
    # if self.verbose: print('leaky_relu(conv)')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    # x = self.batchnorm2(x, training=training)
    # if self.verbose: print('batchnorm')
    # if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None) 

    x_shape = x.shape.as_list()
    x = tf.reshape(x, shape=(-1, np.prod(x_shape[1:])))
    if self.verbose: print('reshape')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)
    
    if conditional is not None:

        if self.verbose: print('conditional:', conditional, conditional.shape)
        x = tf.concat([x, conditional], 1)

        if self.verbose: print('added_conditional')
        if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)   

    x = self.fc0(x)
    if self.verbose: print('fully_connected')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.nn.leaky_relu(x)
    if self.verbose: print('leaky_relu')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.flatten(x)
    if self.verbose: print('flatten')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.fc1(x)
    if self.verbose: print('fully_connected')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.nn.leaky_relu(x)
    if self.verbose: print('leaky_relu')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.fc2(x)
    if self.verbose: print('fully_connected')
    if self.verbose: print('x.shape', x.shape if not isinstance(x, list) else None)

    return x

class GAN(object):
    
    def __init__(self, noise_dim = 100, data_format = 'channels_first', verbose = False):
        
        self.data_format   = data_format
        self.noise_dim     = noise_dim
        self.generator     = Generator    (data_format = data_format, verbose = verbose)
        self.discriminator = Discriminator(data_format = data_format, verbose = verbose)

    
if __name__ == '__main__':
    
    import numpy as np

    tf.enable_eager_execution()
    gan           = GAN(data_format = 'channels_last', verbose = True) 
    generator     = gan.generator
    discriminator = gan.discriminator

    # Defun gives 10 secs/epoch performance boost
    generator.call     = tf.contrib.eager.defun(generator.call)
    discriminator.call = tf.contrib.eager.defun(discriminator.call)

    num_examples_to_generate     = 4
    random_vector_for_generation = tf.random_normal([num_examples_to_generate, 3, gan.noise_dim]) if gan.data_format == 'channels_first' else tf.random_normal([num_examples_to_generate, gan.noise_dim, 3]) 

    if True:
        conditional = np.random.normal(size = (num_examples_to_generate, 3)).astype(np.float32)
    else:
        conditional = None
    # 
    gen = generator(random_vector_for_generation, conditional = conditional)
    print()
    print()
    dis = discriminator(gen, conditional = conditional)
    print()

    # Sun total variables.
    gvn = np.sum([np.prod(v.shape.as_list()) for v in generator.variables])
    dvn = np.sum([np.prod(v.shape.as_list()) for v in discriminator.variables])

    print('Total Gen Vars:', gvn)
    print('Total Dis Vars:', dvn)