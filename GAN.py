import tensorflow as tf

class Generator(tf.keras.Model):
  def __init__(self, data_format = 'channels_first'):
    super(Generator, self).__init__()
    
    self.data_format = data_format
    
    # Define layers.
    self.flatten    = tf.keras.layers.Flatten()
    self.fc1        = tf.keras.layers.Dense(500, use_bias = False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization(axis = 2 if self.data_format == 'channels_last' else 1)
    
    self.conv1 = tf.keras.layers.UpSampling1D(size = 3)
    self.batchnorm2 = tf.keras.layers.BatchNormalization(axis = 2 if self.data_format == 'channels_last' else 1)
    
    self.conv2 = tf.keras.layers.UpSampling1D(size = 3)
    self.batchnorm3 = tf.keras.layers.BatchNormalization(axis = 2 if self.data_format == 'channels_last' else 1)
    
    self.conv3 = tf.keras.layers.UpSampling1D(size = 2)

  def call(self, x, training = True):
    
    # print('Generator.call()', training)
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.transpose(x, perm = [0, 2, 1]) if self.data_format == 'channels_last' else x
    # print('reshape')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.fc1(x)
    # print('fc1')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.batchnorm1(x, training=training)
    # print('batchnorm')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.relu(x)
    # print('relu')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.reshape(x, shape=(-1, 500, 3))
    # print('reshape')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.conv1(x)
    # print('conv1')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.batchnorm2(x, training = training)
    # print('batchnorm')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.relu(x)
    # print('relu')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.conv2(x)
    # print('conv')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.batchnorm3(x, training = training)
    # print('batchnorm')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.relu(x)
    # print('relu')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    # x = tf.nn.tanh( self.conv3(x) ) 
    # print('tanh(conv)')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.transpose(x, perm = [0, 2, 1]) if self.data_format == 'channels_first' else x
    # print('reshape')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    return x

class Discriminator(tf.keras.Model):
  def __init__(self, data_format = 'channels_first'):
    super(Discriminator, self).__init__()
    
    # Define layers. 
    self.conv1   = tf.keras.layers.Conv1D(3, 5, strides = 3, padding = 'same', data_format = data_format)
    self.conv2   = tf.keras.layers.Conv1D(3, 5, strides = 3, padding = 'same', data_format = data_format)
    self.conv3   = tf.keras.layers.Conv1D(3, 5, strides = 2, padding = 'same', data_format = data_format) 
    self.dropout = tf.keras.layers.Dropout(0.3)
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(1)

  def call(self, x, training = True):
    
    # print('Discriminator.call()', training)
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.nn.leaky_relu(self.conv1(x))
    # print('leaky_relu(conv)')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.dropout(x, training=training)
    # print('droupout')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = tf.nn.leaky_relu(self.conv2(x))
    # print('leaky_relu(conv)')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.dropout(x, training=training)
    # print('droupout')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    x = tf.nn.leaky_relu(self.conv3(x))
    # print('leaky_relu(conv)')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.dropout(x, training=training)
    # print('droupout')
    # print('x.shape', x.shape if not isinstance(x, list) else None)

    x = self.flatten(x)
    # print('flatten')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
    
    x = self.fc1(x)
    # print('fully_connected')
    # print('x.shape', x.shape if not isinstance(x, list) else None)
   
    return x

class GAN(object):
    
    def __init__(self, noise_dim = 100, data_format = 'channels_first'):
        
        self.data_format   = data_format
        self.noise_dim     = noise_dim
        self.generator     = Generator    (data_format = data_format)
        self.discriminator = Discriminator(data_format = data_format)