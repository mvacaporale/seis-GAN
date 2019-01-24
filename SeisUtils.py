# SeisUtils.py

'''
Data extraction utilities. 
'''
DATA_PATH_U = 'wformMat_jpm4p_181106.h5'

import h5py
from   keras.utils import Sequence
import numpy as np

#-----------------------------
# Data Extractors
#-----------------------------

# Divide data into train and test set.
def extract_func(burn_seconds = 5, pred_seconds = 10, data_format = 'channels_first'):
    
    def wrapper(data):
        
        # Transpose the data.
        data = data.transpose((2,0,1))
  
        # Take first part of wave form after the burn-in period for input. Take the max of the remaining sequence for output.
        iburn   = burn_seconds * 100 - 1
        itime   = pred_seconds * 100 + iburn 
        x       = data[:, :, iburn:itime ]
        y       = data[:, :, itime:-1    ]
        
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
        self.meta        = self.f['numMeta'];
        self.wform_names = self.f['wformFullNames'];
        self.wform       = self.f['wforms'];

        # Define metad.
        self.metad = {
            'dist' : self.meta[ 1,:],
            'magn' : self.meta[ 2,:],
        }

class SeisGenerator(Sequence):

    def __init__(self, data, batch_size, extract_Xy, metad = None, indeces = None, weights = None, shuffle = True, expend = False ):
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
        print('Getting   batch {}'.format(batch))
        indeces      = self.__indeces[batch * self.batch_size : min([(batch+1) * self.batch_size, len(self.__indeces) ])]
        batch_o_data = self.__get_indeces(indeces, include_weight = self.weights is not None, include_meta = self.metad is not None )
        print('Retreived batch {}'.format(batch))
        return batch_o_data
    
    def __get_indeces(self, indeces, include_weight = True, include_meta = False):
        '''
        Get data over specified indeces
        '''
        X, y = self.extract_Xy(self.data[:, :, indeces])
        data = [X, y]
        if include_weight: data.append(self.weights[indeces])
        if include_meta  : data.append(np.column_stack([ meta[indeces] for k, meta in self.metad.items() ]))
 
        return tuple(data)
    
    def get_indeces(self, indeces, include_weight = True, include_meta = False):
        '''
        Helper function. May be used to get validation and test sets.
        '''
        return self.__get_indeces(indeces, include_weight = include_weight, include_meta = include_meta)
    
    def get_targets(self):
        
        return self.__targets, self.__targetIdx
        
    def on_epic_end(self):
        '''
        Called after each epic of training.
        '''    
        if self.shuffle: np.random.shuffle(self.__indeces)

if __name__ == '__main__':
    
    # Load data.
    data = SeisData()

    # Print data attributes.
    print()
    print('Loading data...')
    print('wform.shape ', data.wform.shape         )
    print('wform.chunks', data.wform.chunks        )
    print('dist .shape ', data.metad['dist'].shape )
    print('magn .shape ', data.metad['magn'].shape )

    # Test extract_func.
    f = extract_func(data_format = 'channels_last')
    # f = extract_func(data_format = 'channels_first')
    print()
    print('Extracting data (\'channels_last\')...')
    w = data.wform[:, :, 1:10]
    print('in  shape:', w.shape      )
    print('out shape:', f(w)[0].shape)

