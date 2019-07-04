import numpy     as np
import tensorflow as tf

from GAN             import GAN
from SeisUtils       import extract_func, Transformation
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

        def make_bins(metav, bins):

            if isinstance(bins, int):
                b     = np.linspace(np.min(metav), np.max(metav), bins + 1)
                b[-1] = np.inf
                return b
            elif isinstance(bins, list):
                b     = np.array([np.inf if b_ == 'inf' else b_ for b_ in bins])
                return b
            else:
                assert False, "Bins must either be specified as a list or an interger."


        self.data        = SeisData(data_path = config['data_path'])
        f                = self.data.f            # h5 dictionary
        meta             = self.data.meta         # meta info (13, 260764) includes 'dist' and 'magn'
        wform_names      = self.data.wform_names  # names (e.g '/proj..japan...446.UD')
        wform            = self.data.wform        # wave forms - (3, 10500, 260764)
        metad            = self.data.metad        # { 'dist' : [...], 'magn' : [...] } (distance and magnitude dict for easier access)

        # Fix random seed for reproducibility
        np.random.seed(self.random_seed)

        # Define training, testing, and validation indices.
        train_frac = self.config.get('train_frac') or 0.74
        test_frac  = (0.2 / 0.36) * (1 - train_frac)
        valid_frac = 1 - train_frac - test_frac
        assert train_frac + test_frac + valid_frac == 1.0, "Training set not of valid size {}. All sets must add up to 100%".format(train_frac)

        # Define training, testing, and validation indices.
        N                  = wform.shape[2] 
        n1, n2             = int(np.ceil(test_frac * N)), int(np.ceil(valid_frac * N))  
        all_indices        = np.random.choice(N, N, replace = False)
        testIdx, validIdx, trainIdx = np.split(all_indices, [n1, n1 + n2])
        testIdx.sort(), validIdx.sort(), trainIdx.sort();
        if config.get('debug'): print('Testing    Samples:', len(testIdx ))
        if config.get('debug'): print('Validation Samples:', len(validIdx))
        if config.get('debug'): print('Training   Samples:', len(trainIdx), '\n')

        # Save which samples have been selected for training.
        self.train_indeces = trainIdx

        # Set default weights and conditionals.
        self.weights           = None 
        self.conditional_metav = None

        # Make list of data array to include as conditional data.
        self.metas = [metad[cond] for cond in self.config.get('metas', [])]

        # Difine bins for meta data if needed.
        if 'bins_s' in self.config:

            # Assert metas have been given.
            assert 'metas' in self.config, 'Missing \'metas\' from config: List of keys to metas must be defined to construct bins.'
            
            # Construct bins.
            self.bins  = [make_bins(metad[meta], bins) for meta, bins in zip(self.config['metas'], self.config['bins_s'])]

            # Cpnstruct hist from bins.
            self.H, _     = np.histogramdd(self.metas, bins = self.bins)
            self.H_scaled = np.divide(1, self.H, out = np.zeros_like(self.H).astype(np.float32), where = self.H != 0)
            self.H_scaled = self.H_scaled / (np.sum(self.H_scaled))

            # Define corresponding weights.
            self.cond_idxs = tuple([tuple(np.digitize(m, b) - 1) for m, b in zip(self.metas, self.bins)])
            self.weights   = self.H_scaled[self.cond_idxs]
            self.weights   = self.weights / np.mean(self.weights)

            # Transform meta data for graph compatible format.
            if self.config.get('conditional_config', {}).get('one_hot_encode', False):

                # Assert bins have been defined.
                assert 'bins_s' in self.config, 'Missing \'bins_s\' from config: List of bins as ints or arrays must be given to one hot encode conditional data.'

                # One hot encode conditional data according to bins.
                conditionals = []
                for cond_idx, b in zip(self.cond_idxs, self.bins):
                    conditional = np.zeros((len(cond_idx), len(b) - 1))
                    conditional[np.arange(len(cond_idx)), cond_idx]  = 1
                    conditionals.append(conditional)
                self.conditional_metav = conditionals # np.concatenate(conditionals, 1)

                # Decide whether to train auxiliary classifier. 
                if self.config.get('conditional_config', {}).get('aux_classify', False):

                    self.config['aux_classify']         = True
                    self.config['aux_categories_sizes'] = [c.shape[1] for c in self.conditional_metav]

            elif self.config.get('conditional_config', {}):

                # Assert metas have been given
                assert 'metas' in self.config, 'Missing \'metas\' from config: List of keys to metas must be defined to construct normalized bins.'

                # Normalize conditional values to lie between 0 and 1.
                metad_filt             = {meta_key : metad[meta_key] / np.max(metad[meta_key]) for meta_key in self.metas}
                self.conditional_metav = np.column_stack([ meta for k, meta in metad_filt.items() ]) if metad_filt else None

        # Print debug.
        if config.get('debug'): print('conditional_metav (shape):', [m.shape for m in self.conditional_metav] if isinstance(self.conditional_metav, list) else self.conditional_metav.shape if self.conditional_metav is not None else None)
        if config.get('debug'):
            if self.weights is not None:
                print('weights:', self.weights)
                print('   max  = ', np.max(self.weights))
                print('   min  = ', np.min(self.weights))
                print('   mean = ', np.mean(self.weights))
                print('   std  = ', np.std(self.weights), '\n')
            else:
                print('weights', None)

        # Get transformation.
        t_name   = config.get('transformation_name', None)
        t_params = config.get('transformation_paramd', {})

        self.transformation   = Transformation.classd[t_name](t_params) if t_name else None

        # Define data generator.
        # weights     = 10 * (1 / metad['dist']) * np.min(metad['dist']).astype(np.float32) if self.config.get('weight_loss', False) else None
        self.extract_f   = extract_func(
            data_format    = self.data_format,
            burn_seconds   = config['burn_seconds'], 
            input_seconds  = config['input_seconds'],
            output_seconds = config['output_seconds'],
            measure_rate   = config['measure_rate'],
            normalize      = config.get('normalize_data', True), 
            transform      = self.transformation.transform if self.transformation else None
        )
        self.SG_test     = SeisGenerator(
            wform, 
            self.batch_size, 
            self.extract_f, 
            metav   = self.conditional_metav, 
            indices = testIdx , 
            verbose = True, 
            shuffle = False, 
            expend  = True)
        self.SG_valid    = SeisGenerator(
            wform, 
            self.batch_size, 
            self.extract_f, 
            metav = self.conditional_metav, 
            weights = self.weights, 
            indices = validIdx, 
            verbose = True, 
            shuffle = True)
        self.SG_train    = SeisGenerator(
            wform, 
            self.batch_size, 
            self.extract_f, 
            metav = self.conditional_metav, 
            weights = self.weights, 
            indices = trainIdx, 
            verbose = True, 
            shuffle = True, 
            normalizers = self.config.get('normalizers'),
            load_data   = self.config.get('load_data_into_memory', False))

        # print(self.config.get('normalizers')
        self.config['normalizers'] = self.SG_train.normalizers

        if config.get('debug'): print('Testing    Samples:', len(self.SG_test ))
        if config.get('debug'): print('Validation Samples:', len(self.SG_valid))
        if config.get('debug'): print('Training   Samples:', len(self.SG_train))

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