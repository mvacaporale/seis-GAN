# SeisUtils.py

'''
Data extraction utilities. 
'''
DATA_PATH_U = 'wformMat_jpm4p_181106.h5'

import h5py
import numpy as np
from   keras.utils import Sequence
from   collections import namedtuple

#-----------------------------
# Data Extractors
#-----------------------------

# Divide data into train and test set.
def extract_func(burn_seconds = 5, input_seconds = 10, output_seconds = None, measure_rate = 100, data_format = 'channels_first', normalize = True):
    
    def wrapper(data):

  
        # Take first part of wave form after the burn-in period for input. Take the max of the remaining sequence for output.
        iburn   = int(burn_seconds   * measure_rate - 1    )
        itime1  = int(input_seconds  * measure_rate + iburn)
        itime2  = int(output_seconds * measure_rate + itime1 if output_seconds is not None else -1)
        x       = data[:, iburn:itime1 , : ]
        y       = data[:, itime1:itime2, : ]
        
        x = x / np.max(np.abs(x), axis = 1)[:, None, :]  
        y = y / np.max(np.abs(y), axis = 1)[:, None, :]  
        
        x = x.transpose((2,0,1))
        y = y.transpose((2,0,1))
        

        return (x, y) if data_format == 'channels_first' else (x.transpose((0, 2, 1)), y.transpose((0, 2, 1)))
    
    return wrapper
    
#-----------------------------
# Data Handlers
#-----------------------------

class SeisData(object):
    '''
    Handles loading of data. Contains easy access attributes. 
    '''

    def __init__(self, data_path = DATA_PATH_U):

        # Load h5. 
        self.f = h5py.File(data_path, 'r')

        # Extract wave forms.
        self.meta        = self.f['numMeta']; # Event magnitude / Hypocentral distance [km] / Earthquake hypocenter depth  [km] / Signal/noise power ratio / Station latitude / Station longitude / Earthquake latitude / Earthquake longitude / back-azimuth / Earthquake origin time / Unique record ID / Unique earthquake ID / Onset index 
        self.wform_names = self.f['wformFullNames'];
        self.wform       = self.f['wforms'];

        # Define metad.
        self.metad = {
            'magn' : self.meta[ 0,:],
            'dist' : self.meta[ 1,:],
        }

class SeisGenerator(Sequence):

    # Define named tuple to return from batch extraction. 
    batch = namedtuple('batch', 'x y weights metav' )
    batch.__new__.__defaults__ = (None,) * len(batch._fields) # Set defaults to None

    def __init__(self, data, batch_size, extract_Xy, metad = None, indeces = None, weights = None, shuffle = True, expend = False, verbose = False ):
        '''
        :var  save_targets: Array to save target y values as they are extracted. Helpful for evaulating test set. 
        :type save_targets: None or numpy array
        :var  expend      : Decides whether to use all the data even if it yields a partially filled final batch
        :type expend      : bool
        '''
        self.data           = data
        self.batch_size     = batch_size
        self.extract_Xy     = extract_Xy
        self.metad          = metad 
        self.weights        = weights
        self.shuffle        = shuffle
        self.expend         = expend 
        self.verbose        = verbose
        self.__indeces      = indeces if indeces is not None else np.arange(self.data.shape[2]) # Subset to use for training.

        
    def __len__(self):
        '''
        Denote the number of batches per epoch.
        '''
        
        return int(np.floor(len(self.__indeces) / self.batch_size)) if not self.expend else int(np.ceil(len(self.__indeces) / self.batch_size))
    
    def __getitem__(self, batch):
        '''
        Create one batch of the data.
        '''
        if self.verbose: print('Getting   batch {}'.format(batch))
        indeces       = self.__get_indeces(batch)
        batch_o_items = self.__get_data(indeces, include_weight = self.weights is not None, include_meta = self.metad is not None )
        if self.verbose: print('Retreived batch {}'.format(batch))
        return batch_o_items

    def __get_indeces(self, batch):

        return self.__indeces[batch * self.batch_size : min([(batch+1) * self.batch_size, len(self.__indeces) ])]

    def __get_meta(self, indeces):

        return np.column_stack([ meta[indeces] for k, meta in self.metad.items() ]) if self.metad else None

    def __get_data(self, indeces, include_weight = True, include_meta = False):
        '''
        Get data over specified indeces
        '''
        X, y = self.extract_Xy(self.data[:, :, indeces])
        data = [X, y]
        weights = self.weights[indeces]    if self.weights is not None else None
        metav   = self.__get_meta(indeces) if self.metad               else None
 
        return self.batch(X, y, weights, metav)
    
    def get_indeces(self, indeces, include_weight = True, include_meta = False):
        '''
        Helper function. May be used to get validation and test sets.
        '''
        return self.__get_data(indeces, include_weight = include_weight, include_meta = include_meta)
    
    def random_batch(self):

        return self.__getitem__(np.random.randint(0, self.__len__()))

    def random_metav(self):

        batch = np.random.randint(0, self.__len__())
        if self.verbose: print('Getting   metav {}'.format(batch))
        indeces      = self.__get_indeces(batch)
        metav        = self.__get_meta(indeces)
        weights      = self.weights[indeces]    if self.weights is not None else None
        if self.verbose: print('Retreived metav {}'.format(batch))

        return self.batch(metav = metav, weights = weights)

    def get_targets(self):
        
        return self.__targets, self.__targetIdx
        
    def on_epic_end(self):
        '''
        Called after each epic of training.
        '''    
        if self.shuffle: np.random.shuffle(self.__indeces)

if __name__ == '__main__':
    
    # Load data.
    data = SeisData(data_path = 'wformMat_jpm4p_181106_downsample-5x.h5')

    # Print data attributes.
    print()
    print('Loading data...')
    print('wform.shape ', data.wform.shape         )
    print('wform.chunks', data.wform.chunks        )
    print('dist .shape ', data.metad['dist'].shape )
    print('magn .shape ', data.metad['magn'].shape )

    # Test extract_func.
    f = extract_func(data_format = 'channels_last', burn_seconds = 2.5, input_seconds = 20, measure_rate = 20)
    # f = extract_func(data_format = 'channels_first')
    print()
    print('Extracting data (\'channels_last\')...')
    w = data.wform[:, :, 1:10]
    print('in  shape:', w.shape      )
    print('out shape x, y:', f(w)[0].shape, f(w)[1].shape)

    weights     = (1 / data.metad['dist']) * np.min(data.metad['dist'])
    SG_train    = SeisGenerator(data.wform, 3, f, metad = {'dist' : data.metad['dist'], 'magn' : data.metad['magn']}, indeces = range(10000), verbose = True, shuffle = True)

    print(SG_train.random_metav().shape)