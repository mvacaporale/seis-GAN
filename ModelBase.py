import numpy     as np
import tensorflow as tf
import matplotlib.pyplot as plt

from GAN             import GAN
from SeisUtils       import extract_func
from SeisUtils       import SeisData
from SeisUtils       import SeisGenerator
from GAN             import GAN

import os
import copy 
import json

class ModelBase(object):
    
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
            
        # Deepcopy to avoid any potential mutation.
        self.config = copy.deepcopy(config)

        # Print contents.
        if self.config.get('debug'): 
            print('config', self.config)

        # Store random seed to set with tf.set_random_seed().
        self.random_seed = self.config['random_seed']

        # Save training params.
        self.epochs      = self.config['epochs']
        self.batch_size  = self.config['batch_size']
        self.data_format = self.config['data_format']
        
        # Init save directory and data.
        self._init_save_directory()
        self._init_data(config)
        
        # Init child Model custom parameters.
        self.set_agent_props()

        # Build/make graph, session, saver, and file writer.
        self.build_graph_and_session()
          
        # Initialize global variables or restore checkpoint.
        if self.config.get('restore'): 
            # Restore gloabel variables.
            print('Loading the model from folder: %s' % self.checkpoint_dir)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))
        else:
            # Initialize global variables.
            tf.set_random_seed(self.random_seed)
            self.sess.run(self.init_op)   
        
        # Save config.
        with open(os.path.join(self.project_dir, 'config.json'), 'w') as config_file:
            json.dump(self.config, config_file)
        
    def _init_save_directory(self):
        
        self.project_dir     = self.config.get('directory') or './GAN_{}'.format(datetime.datetime.now().date().isoformat())
        self.checkpoint_dir  = os.path.join(self.project_dir, './checkpoints')
        self.images_dir      = os.path.join(self.project_dir, './images'     )
        self.diagnostics_dir = os.path.join(self.project_dir, './diagnostics')
        self.globalstep_dir  = os.path.join(self.project_dir, './globalsteps')

        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)    
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(self.diagnostics_dir):
            os.makedirs(self.diagnostics_dir)
        if not os.path.exists(self.globalstep_dir):
            os.makedirs(self.globalstep_dir)    
            
    def _init_data(self, config):

        self.data        = SeisData(data_path = config['data_path'])
        f                = self.data.f            # h5 dictionary
        meta             = self.data.meta         # meta info (13, 260764) includes 'dist' and 'magn'
        wform_names      = self.data.wform_names  # names (e.g '/proj..japan...446.UD')
        wform            = self.data.wform        # wave forms - (3, 10500, 260764)
        metad            = self.data.metad        # { 'dist' : [...], 'magn' : [...] } (distance and magnitude dict for easier access)

        # Fix random seed for reproducibility
        np.random.seed(self.random_seed)

        # Define training, testing, and validation indeces.
        train_frac = self.config.get('train_frac') or 0.74
        test_frac  = (0.2 / 0.36) * (1 - train_frac)
        valid_frac = 1 - train_frac - test_frac
        assert train_frac + test_frac + valid_frac == 1.0, "Training set not of valid size {}. All sets must add up to 100%".format(train_frac)

        # Define training, testing, and validation indeces.
        N                  = wform.shape[2] 
        n1, n2             = int(np.ceil(test_frac * N)), int(np.ceil(valid_frac * N))  
        all_indeces        = np.random.choice(N, N, replace = False)
        testIdx, validIdx, trainIdx = np.split(all_indeces, [n1, n1 + n2])
        testIdx.sort(), validIdx.sort(), trainIdx.sort();
        if config.get('debug'): print('Testing    Samples:', len(testIdx ))
        if config.get('debug'): print('Validation Samples:', len(validIdx))
        if config.get('debug'): print('Training   Samples:', len(trainIdx))

        # Define data generator.
        weights     = 10 * (1 / metad['dist']) * np.min(metad['dist']).astype(np.float32) if self.config.get('weight_loss', False) else None
        metad_filt  = {meta_key : metad[meta_key] for meta_key in config.get('metas', [])}
        self.extract_f   = extract_func(
            data_format    = self.data_format,
            burn_seconds   = config['burn_seconds'], 
            input_seconds  = config['input_seconds'],
            output_seconds = config['output_seconds'],
            measure_rate   = config['measure_rate']
        )
        self.SG_test     = SeisGenerator(
            wform, 
            self.batch_size, 
            self.extract_f, 
            metad   = metad_filt, 
            indeces = testIdx , 
            verbose = True, 
            shuffle = False, 
            expend  = True)
        self.SG_valid    = SeisGenerator(wform, self.batch_size, self.extract_f, metad = metad_filt, weights = weights, indeces = validIdx, verbose = True, shuffle = True)
        self.SG_train    = SeisGenerator(wform, self.batch_size, self.extract_f, metad = metad_filt, weights = weights, indeces = trainIdx, verbose = True, shuffle = True)
        if config.get('debug'): print('Testing    Samples:', len(self.SG_test ))
        if config.get('debug'): print('Validation Samples:', len(self.SG_valid))
        if config.get('debug'): print('Training   Samples:', len(self.SG_train))

        # Load all the data in memory.
        if config.get('load_data_into_memory'):
            print('Loading data...')
            self.SG_train.data = self.SG_train.data[:]
            print('Loaded...\n')

    def set_agent_props(self):
        pass
    
    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden.')

    @staticmethod
    def get_random_config(fixed_params={}):
        # Generate random configuration of the current model.
        raise Exception('The get_random_config function must be overriden.')

    def infer(self):
        raise Exception('The infer function must be overriden.')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden.')
    
    def plot_wforms(self, w, figsize = (40,10), ylabels = None):
        if isinstance(w, tf.Tensor): w = w.numpy()
        N = w.shape[0] if len(w.shape) > 2 else 1
        fig, axs = plt.subplots(N, 3, figsize = figsize)
        for i in range(N):
            for j in range(3):
                ax = axs[i, j] if N > 1 else axs[j]
                if N > 1:
                    ax.plot(w[i, :, j]) 
                else:
                    ax.plot(w[:, j])
        for i in range(N):
            ax = axs[i, 0] if N > 1 else axs[0]
            ax.set_ylabel(ylabels[i] if ylabels is not None else 'velocity')
        for j in range(3):
            ax = axs[N-1, j] if N > 1 else axs[j]
            ax.set_xlabel('channel {}'.format(j + 1))
        return fig

if __name__ == '__main__':
    
    config_d = {
    
        #---------
        # Data
        #---------
        
        'data_path'   : 'wformMat_jpm4p_181106_downsample-5x.h5', # path to data
        'data_format' : 'channels_last', # data format ('channels_first' or 'channels_last')
        'frac_train'  : 0.74,  # % of data devoted to Training
        
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
        
        'batch_size'  : 4, # batch size
        'epochs'      : 1, # training epochs
        'random_seed' : 7, # random seed
        'metas'       : ['dist', 'magn'], # meta to include as conditional parameters.
        
        #--------
        # Saving
        #--------
        
        'directory'   : 'TEST',# defaults to todays date. Will make a dir called 'GAN_<directory>
        'restore'     : False,
        
        #--------
        # Debug
        #--------

        'debug' : True,

    }

    ModelBase(config_d)