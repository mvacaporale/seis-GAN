# Import modules.
import os
import matplotlib.pyplot as mp

# Import tools for keras.
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from librosa.display import waveplot
from IPython import display
from keras.utils import OrderedEnqueuer

import datetime
import time
import argparse

from SeisUtils import extract_func
from SeisUtils import SeisData
from SeisUtils import SeisGenerator
from GAN       import GAN
from ModelBase import ModelBase

# Parse arguments.
parser = argparse.ArgumentParser( 
    description = '''
    Run GAN on seismic waveforms. 
    ''')
# parser.add_argument( '-r', '--Restore'   , action = 'store_true', help = "Restore - whether to restore from latest checkpoint - default = False."               )
parser.add_argument( '-t', '--TrainFrac'  , type   = float       , help = "Fraction of data set to use for training - default 0.74"              , default = 0.74)
parser.add_argument( '-d', '--ProjectDir' , type   = str         , help = "Name of project directory. default = <today's iso date>"              , default = None)
parser.add_argument( '-v', '--Verbose'    , type   = bool        , help = "Whether to print verbose feedback."                                   , default = True)
args = parser.parse_args() 
print('Restore:', args.ProjectDir, bool(args.ProjectDir))


from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print(get_available_gpus())
print()

config_d = {
    
    #---------
    # Data
    #---------
    
    'data_path'   : '/seis/wformMat_jpm4p_181106_downsample-5x.h5', # path to data
    'data_format' : 'channels_last', # data format ('channels_first' or 'channels_last')
    'train_frac'  : args.TrainFrac,  # % of data devoted to Training
    
    #---------
    # Wforms
    #---------

    'burn_seconds'   : 2.5,  # first  part of wform to throw away
    'input_seconds'  : 20,   # middle part of waveform to use as input
    'output_seconds' : None, # last   part of waveform to use as target output or None if generting x.
    'measure_rate'   : 20,   # sampling rate in HZ

    #---------
    # Training
    #---------
    
    'batch_size'  : 256, # batch size
    'epochs'      : 1000, # training epochs
    'random_seed' : 7, # random seed
    'load_data_into_memory' : True,
    'metas'       : ['dist', 'magn'], # meta to include as conditional parameters.
    'weight_loss' : True,

    #--------
    # Saving
    #--------
    
    'directory'   : args.ProjectDir or './GAN_{}'.format(datetime.datetime.now().date().isoformat()), # defaults to todays date. Will make a dir called 'GAN_<directory>
    'restore'     : bool(args.ProjectDir),
    
    #--------
    # Debug
    #--------

    'debug' : True,

}

class GANModelBase(ModelBase):
    
    def __init__(self, config):
        
        super(GANModelBase, self).__init__(config)

    def noise(self):
        # Generate noise from a uniform distribution
        return np.random.normal(size = [self.batch_size, 3, self.gan.noise_dim]).astype(dtype = np.float32) if self.data_format == 'channel_first' else np.random.normal(size = [self.batch_size, self.gan.noise_dim, 3]).astype(dtype = np.float32)

    def build_graph_and_session(self):
        
        self.graph = tf.Graph()
        with self.graph.as_default():
        
            self.gan           = GAN(data_format = self.data_format, verbose = True) 
            self.generator     = self.gan.generator
            self.discriminator = self.gan.discriminator


            with tf.device('/device:GPU:0'):

                discriminator_optimizer = tf.train.AdamOptimizer(1e-4, beta1 = 0, beta2 = 0.9)
                generator_optimizer     = tf.train.AdamOptimizer(1e-4, beta1 = 0, beta2 = 0.9)
                self.global_step = tf.train.get_or_create_global_step()
                
                # Setup conditional placeholder.
                if self.config.get('metas', []):
                    self.conditional_placeholder = tf.placeholder(tf.float32, shape = [None, len(self.config['metas'])], name = 'conditional') 
                else:
                    self.conditional_placeholder = None
                    
                # Setup loss weight tensor.
                if self.config.get('weight_loss', False):
                    self.weights_placeholder = tf.placeholder(tf.float32, shape = [None, 1], name = 'weights') 
                else:
                    self.weights_placeholder = None
                    
                # Set up input and output placeholders.
                self.gen_input_placeholder      = tf.placeholder(tf.float32, shape = [None, 100, 3], name = 'latent')
                self.gen_output_tensor          = self.generator(self.gen_input_placeholder, conditional = self.conditional_placeholder)
                self.dis_input_real_placeholder = tf.placeholder(tf.float32, shape = self.gen_output_tensor.shape, name = 'real_input')
                self.dis_output_real_tensor     = self.discriminator(self.dis_input_real_placeholder, conditional = self.conditional_placeholder)
                self.dis_output_fake_tensor     = self.discriminator(self.gen_output_tensor         , conditional = self.conditional_placeholder)
                
                #-------------------
                # Generator Step
                #--------------------
                
                # Weight the loss tensor.
                if self.weights_placeholder is not None:
                    dis_output_fake_tensor_weighted = tf.multiply(self.dis_output_fake_tensor, self.weights_placeholder)
                else:
                    dis_output_fake_tensor_weighted = self.dis_output_fake_tensor

                # Define gen loss tensor.
                self.gen_loss_tensor = tf.multiply(tf.constant(-1, dtype = tf.float32), tf.reduce_mean(dis_output_fake_tensor_weighted))
                 
                # Define gradient step tensor. 
                self.gen_calc_grads_op  = tf.gradients(self.gen_loss_tensor, self.generator.variables)
                self.gen_apply_grads_op = generator_optimizer.apply_gradients(zip(self.gen_calc_grads_op, self.generator.variables), global_step = self.global_step)

                #-------------------
                # Discriminator Step
                #--------------------

                # Define dis loss tensor.
                epsilon      = tf.random_uniform([], 0, 1)
                mixed_input  = (epsilon * self.dis_input_real_placeholder) + (1 - epsilon) * self.gen_output_tensor
                mixed_output = self.discriminator(mixed_input, training = True, conditional = self.conditional_placeholder)

                # Compute gradient of mixed_output with respect to it's input - used for regularization.
                mix_calc_grads_op  = tf.gradients(mixed_output, mixed_input)
                norm_of_mixed      = tf.norm(mix_calc_grads_op)
                
                # Weight the logits.           
                if self.weights_placeholder is not None:
                    dis_output_fake_tensor_weighted = tf.multiply(self.dis_output_fake_tensor, self.weights_placeholder)
                    dis_output_real_tensor_weighted = tf.multiply(self.dis_output_real_tensor, self.weights_placeholder)
                else:
                    dis_output_fake_tensor_weighted = self.dis_output_fake_tensor
                    dis_output_real_tensor_weighted = self.dis_output_real_tensor
                                                                
                # Defined loss tensor.
                regularizer = 10 * tf.square(norm_of_mixed - 1)        
                self.dis_loss_tensor = tf.reduce_mean(dis_output_fake_tensor_weighted - dis_output_real_tensor_weighted + regularizer)
                
                # Define gradient step tensor. 
                self.dis_calc_grads_op  = tf.gradients(self.dis_loss_tensor, self.discriminator.variables)
                self.dis_apply_grads_op = discriminator_optimizer.apply_gradients(zip(self.dis_calc_grads_op, self.discriminator.variables))
                
                #-----------------
                # Diagnostics
                #-----------------

                def tensor_norms(tensors):
                    if isinstance(tensors, list):
                        return [tf.norm(g) for g in tensors if g is not None]
                    else:
                        return tf.map_fn(lambda g: tf.norm(g), tensors)

                gen_grads_norm = tf.reduce_mean(tensor_norms(self.gen_calc_grads_op))
                dis_grads_norm = tf.reduce_mean(tensor_norms(self.dis_calc_grads_op))
                mix_grads_norm = tf.reduce_mean(tensor_norms(mix_calc_grads_op))

                real_logits  = tf.reduce_mean(self.dis_output_real_tensor)
                fake_logits  = tf.reduce_mean(self.dis_output_fake_tensor)
                mixed_logits = tf.reduce_mean(mixed_output)

            #----------------------
            # Summaries and prints.
            #----------------------
            with tf.name_scope('GradientNorms'):
                # Generator.
                tf.summary.scalar('gen_grads_norm', gen_grads_norm)
                tf.add_to_collection('Prints', tf.Print(gen_grads_norm, [gen_grads_norm], message = ' '*5 + 'gen_grads_norm: '))

                # Discriminator.
                tf.summary.scalar('dis_grads_norm', dis_grads_norm)
                tf.add_to_collection('Prints', tf.Print(dis_grads_norm, [dis_grads_norm], message = ' '*5 + 'dis_grads_norm: '))     

                # Mixed.
                tf.summary.scalar('mix_grads_norm', mix_grads_norm)
                tf.add_to_collection('Prints', tf.Print(mix_grads_norm, [mix_grads_norm], message = ' '*5 + 'mix_grads_norm: '))     

            with tf.name_scope('Losses'):
                # Generator.    
                tf.summary.scalar('gen_losses', self.gen_loss_tensor)
                tf.add_to_collection('Prints', tf.Print(self.gen_loss_tensor, [self.gen_loss_tensor], message = ' '*5 + 'gen_losses: ')) 

                # Discriminator.
                tf.summary.scalar('dis_losses', self.dis_loss_tensor)
                tf.add_to_collection('Prints', tf.Print(self.dis_loss_tensor, [self.dis_loss_tensor], message = ' '*5 + 'dis_losses: '))

                # Regularizer. 
                tf.summary.scalar('regularizer', regularizer)
                tf.add_to_collection('Prints', tf.Print(regularizer, [regularizer], message = ' '*5 + 'regularizer: '))

            with tf.name_scope('Logits'):
                # Real.
                tf.summary.scalar('real_logits', real_logits)
                tf.add_to_collection('Prints', tf.Print(real_logits, [real_logits], message = ' '*5 + 'real_logits: '))

                # Fake.
                tf.summary.scalar('fake_logits', fake_logits)
                tf.add_to_collection('Prints', tf.Print(fake_logits, [fake_logits], message = ' '*5 + 'fake_logits: ')) 

                # Mixed.
                tf.summary.scalar('mixed_logits', mixed_logits)
                tf.add_to_collection('Prints', tf.Print(mixed_logits, [mixed_logits], message = ' '*5 + 'mixed_logits: ')) 

            # Merge summaries and prints.
            self.summary_op   = tf.summary.merge_all()
            self.init_op      = tf.initializers.global_variables()
            self.print_op     = tf.group(*tf.get_collection('Prints'))
            self.saver        = tf.train.Saver()

        # Add all the other common code for the initialization here
        sessConfig  = tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
        self.sess   = tf.Session(config = sessConfig, graph = self.graph)
        self.sw     = tf.summary.FileWriter(self.diagnostics_dir, self.sess.graph)
            
class GANModel(object):
    
    def __init__(self, gm):
    
        self.gm = gm
        
    def __getattribute__(self, name):
        
        if name in ['gm', 'train']:
            return super(GANModel, self).__getattribute__(name)
        else:
            return getattr(self.gm, name)
    
    def train(self, n_critic = 5):  
        
        conditional = False
        data_format = self.data_format
        cat_dim     = 2 if data_format == 'channel_first' else 1
        noise_dim   = self.gan.noise_dim

        # Perform training epochs. 
        for epoch in range(self.epochs):
            start = time.time()

            # Iterate over entire dataset per epoch. 
            for i in range(len(self.SG_train)):

                # Take learning steps for discriminator. 
                for nc in range(n_critic):

                    # Sample from the data.
                    batch     = self.SG_train.random_batch()
                    wform_x   = batch.x
                    
                    feed_dict = {self.gen_input_placeholder : self.noise(), self.dis_input_real_placeholder : wform_x[:, :, None, :]}
                    if self.conditional_placeholder is not None:
                        feed_dict[self.conditional_placeholder] = batch.metav
                    if self.weights_placeholder is not None:
                        feed_dict[self.weights_placeholder]     = batch.weights[:, None]
                        
                    if i % 50 == 1:
                        _, _, summary, step = self.sess.run([self.dis_apply_grads_op, self.print_op, self.summary_op, self.global_step], feed_dict = feed_dict)
                        print('     ... Saving summaries at global step:', step)
                        self.sw.add_summary(summary, step)
                    else:
                        self.sess.run(self.dis_apply_grads_op, feed_dict = feed_dict)

                # Take one learning step for generator.
                batch   = self.SG_train.random_metav()
                
                feed_dict = {self.gen_input_placeholder : self.noise()}
                if self.conditional_placeholder is not None:
                    feed_dict[self.conditional_placeholder] = batch.metav
                if self.weights_placeholder is not None:
                    feed_dict[self.weights_placeholder]     = batch.weights[:, None]
                        
                self.sess.run(self.gen_apply_grads_op, feed_dict = feed_dict)

            # Save the model (checkpoint) every epochs.
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'ganckpt.ckpt'), global_step = self.global_step)
            print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

if __name__ == '__main__':

    t1 = time.time()

    gm_b = GANModelBase(config_d)
    gm   = GANModel(gm_b)
    gm.train()

    print('total elapsed time:', time.time() - t1)
