# SeisUtils.py

'''
Data extraction utilities. 
'''
DATA_PATH_U = 'wformMat_jpm4p_181106.h5'

import h5py
import numpy as np
from   keras.utils import Sequence
from   collections import namedtuple
from   tqdm        import trange
import matplotlib.pyplot as plt
from scipy.special import lambertw

import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth = True)
sessConfig  = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, gpu_options = gpu_options)
sess        = tf.Session(config = sessConfig)

#-----------------------------
# Data Extractors
#-----------------------------

SAFE_LOG_MIN = -1000

class Transformation(object):

    def __init__(self, paramd = {}):
        self.paramd = paramd

    def transform(self, arr_old):
        raise NotImplementedError()

    def retransform(self, arr_old):
        raise NotImplementedError()

    # Class registry: To be used to lookup transformations by name as needed.
    classd = {}
    @classmethod
    def register(cls, name = None):
        name = name or cls.__name__
        Transformation.classd[ name ] = cls

class ExponentialScaling(Transformation):
    
    def __init__(self, paramd = {}):
        super(ExponentialScaling, self).__init__(paramd)

    def transform(self, arr_old):
        arr_new = np.zeros(arr_old.shape)
        arr_new[arr_old > 0] = arr_old[arr_old > 0] / np.exp( arr_old[arr_old > 0] / .5)
        arr_new[arr_old < 0] = arr_old[arr_old < 0] / np.exp(-arr_old[arr_old < 0] / .5)
        return arr_new

    def retransform(self, arr_old, k = 0):
        arr_new = np.zeros(arr_old.shape)
        arr_new[arr_old > 0] = -0.5 * lambertw(-2 * arr_old[arr_old > 0], k = k)
        arr_new[arr_old < 0] =  0.5 * lambertw( 2 * arr_old[arr_old < 0], k = k)
        return arr_new

ExponentialScaling.register('exponential_scaling')

class ArcSinh(Transformation):
    
    def __init__(self, paramd = {}):
        super(ArcSinh, self).__init__(paramd)

    def transform(self, arr_old):
        L = self.paramd.get('pre_scale', 0.01)
        arr_new = np.arcsinh(arr_old / L)
        return arr_new

    def retransform(self, arr_old):
        L = self.paramd.get('pre_scale', 0.01)
        arr_new = np.sinh(arr_old) * L
        return arr_new

ArcSinh.register('arc_sinh')


# Divide data into train and test set.
def extract_func(burn_seconds = 5, input_seconds = 10, output_seconds = None, measure_rate = 100, data_format = 'channels_first', normalize = True, transform = None, training = True):

    def wrapper(data, burn_seconds = burn_seconds, input_seconds = input_seconds, output_seconds = output_seconds, measure_rate = measure_rate, data_format = data_format, normalize = normalize, transform = transform, training = training):

        # Take first part of wave form after the burn-in period for input. Take the max of the remaining sequence for output.
        iburn   = int(burn_seconds   * measure_rate - 1    )
        itime1  = int(input_seconds  * measure_rate + iburn  if input_seconds  is not None else -1)
        itime2  = int(output_seconds * measure_rate + itime1 if output_seconds is not None else -1)
        x       = data[:, iburn:itime1 , : ]
        y       = data[:, itime1:itime2, : ] if input_seconds is not None else None
        
        x_normalization = np.max(np.abs(x), axis = 1)
        y_normalization = np.max(np.abs(y), axis = 1) if input_seconds is not None else None

        if normalize is True:
            x = x / x_normalization[:, None, :]  
            y = y / y_normalization[:, None, :] if input_seconds is not None else None

        x               = x.transpose((2,0,1))
        y               = y.transpose((2,0,1))                         if input_seconds is not None else None
        x_normalization = np.log(x_normalization.transpose() + 1e-300)
        y_normalization = np.log(y_normalization.transpose() + 1e-300) if input_seconds is not None else None

        if not data_format == 'channels_first':
            x = x.transpose((0, 2, 1))
            y = y.transpose((0, 2, 1)) if input_seconds is not None else None

        if transform is not None and training:
            x = transform(x)
            y = transform(y) if input_seconds is not None else None
        
        return (x, y, x_normalization, y_normalization)
    
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
    batch = namedtuple('batch', 'x y weights metav indices x_normalization y_normalization' )
    batch.__new__.__defaults__ = (None,) * len(batch._fields) # Set defaults to None

    def __init__(self, data, batch_size, extract_Xy, metav = None, indices = None, weights = None, shuffle = True, expend = False, verbose = False, load_data = False, normalizers = None):
        '''
        :var  save_targets: Array to save target y values as they are extracted. Helpful for evaulating test set. 
        :type save_targets: None or numpy array
        :var  expend      : Decides whether to use all the data even if it yields a partially filled final batch
        :type expend      : bool
        '''
        self.data                  = data
        self.batch_size            = batch_size
        self.metav                 = metav 
        self.shuffle               = shuffle
        self.expend                = expend 
        self.verbose               = verbose
        self.__indices             = indices if indices is not None else np.arange(self.data.shape[2]) # Subset to use for training.
        self.extract_Xy            = extract_Xy
        self.weights               = weights
        self.p                     = None    # Array of probs to weight sampling.
        self.normalizers           = normalizers

        
        if load_data == 'load' or load_data is True:
            print('Loading data...')
            self.data = self.data[:]

            if self.weights is not None:
                self.weights = self.weights[:][self.__indices]
                if shuffle is not None:
                    self.p = (self.weights / np.sum(self.weights))
            
            print('Loaded...\n')

        if load_data == 'load-and-extract':

            print('Loading data...')
            self.data = self.data
            print('     Loaded.\n')

            print('Extracting data...')
            self.data_extracted = None

            N = len(self.__indices)
            n = self.batch_size * 8
            M = n * int(np.ceil(N / n))
            idxs = (list(reversed(range(N))) + [0] * M)[0:M]

            idxs_by_nA = idxs[::-n]
            idxs_by_nB = idxs[-n::-n]
            assert len(idxs_by_nA) == len(idxs_by_nB) == int(M / n)

            for t in trange(int(M / n)):
                
                iA = idxs_by_nA[t]
                iB = idxs_by_nB[t] + 1
                
                datums = self.extract_Xy(self.__indices[iA:iB])

                # import ipdb; ipdb.set_trace()

                if self.data_extracted is None:
                    self.data_extracted = list([np.zeros(shape = (self.data.shape[-1], ) + d.shape[1:]) for d in datums])
                    self.normalizers    = list([-np.inf] * len(datums))

                for i, d in enumerate(datums):
                    self.data_extracted[i][iA:iB, ...] = d 
                    self.normalizers[i]                = np.max([np.max(np.abs(d)), self.normalizers[i]])

            # data_set = self.extract_Xy(self.__indices) # Get all indices.
            self.extract_Xy = lambda indices: tuple([d[indices, ...] * (1 / n) for d, n in zip(self.data_extracted, self.normalizers)])
            
            # print('     Extracted.\n')
            # data_path = 'tmp.h5' # '/seis/wformMat_jpm4p_181106_downsample-5x_spectrograms.h5'
            # print('Opening files: ...')
            # print('   data_path:', data_path)
            # print()
            # f = h5py.File(data_path, 'w')
            # f2.create_dataset(dset, f1[dset].shape, dtype = f1[dset].dtype)

            print('Loading weights...')
            if self.weights is not None:
                self.weights = self.weights[:][self.__indices]
                if shuffle is not None:
                    self.p = (self.weights / np.sum(self.weights))
            print('     Loaded.\n')
            

    def __len__(self):
        '''
        Denote the number of batches per epoch.
        '''
        
        return int(np.floor(len(self.__indices) / self.batch_size)) if not self.expend else int(np.ceil(len(self.__indices) / self.batch_size))
    
    def __getitem__(self, batch, **kwargs):
        '''
        Create one batch of the data.
        '''
        if self.verbose: print('Getting   batch {}'.format(batch))
        indices       = self.__get_batch(batch)
        batch_o_items = self.__get_data_by_indeces(indices, include_weight = self.weights is not None, include_meta = self.metav is not None, **kwargs)
        if self.verbose: print('Retreived batch {}'.format(batch))
        return batch_o_items

    def __get_batch(self, batch):

        return self.__indices[batch * self.batch_size : min([(batch+1) * self.batch_size, len(self.__indices) ])]

    def __get_random_indices(self):
        self._indices  = self.__indices
        random_indices = np.random.choice(self.__indices, size = np.min([self.batch_size, len(self.__indices)]), p = self.p, replace = False)
        random_indices .sort()
        return random_indices

    def __get_data_by_indeces(self, indices, include_weight = True, include_meta = False, **kwargs):
        '''
        Get data over specified indices
        '''
        X, y, x_normalization, y_normalization = self.extract_Xy(self.data[:, :, indices], **kwargs)
        data = [X, y]
        weights = self.weights[indices]                if self.weights is not None and include_weight else None
        metav   = self.__get_metav_by_indeces(indices) if self.metav   is not None                    else None
 
        return self.batch(X, y, weights, metav, indices, x_normalization, y_normalization)

    def __get_metav_by_indeces(self, indices):

        if isinstance(self.metav, list):
            return [m[indices, :] for m in self.metav]
        else:
            return self.metav[indices, :]
    
    def get_indices(self, indices, include_weight = True, include_meta = False, **kwargs):
        '''
        Helper function. May be used to get validation and test sets.
        '''
        return self.__get_data_by_indeces(indices, include_weight = include_weight, include_meta = include_meta, **kwargs)
    
    def random_batch(self, **kwargs):

        if self.verbose: print('Getting   randomly sampled batch')

        if self.shuffle:
            batch = self.__get_data_by_indeces(self.__get_random_indices(), include_weight = False, **kwargs)
        else:
            batch = self.__getitem__(np.random.randint(0, self.__len__()), **kwargs)
        
        if self.verbose: print('Retreived randomly sampled batch')

        return batch 

    def random_metav(self):

        if self.shuffle:

            if self.verbose: print('Getting   randomly sampled metav.')
            indices      = self.__get_random_indices()
            metav        = self.__get_metav_by_indeces(indices) if self.metav is not None else None
            weights      = None
            if self.verbose: print('Retreived randomly sampled metav.')

        else:    

            if self.verbose: print('Getting   metav {}'.format(batch))
            batch = np.random.randint(0, self.__len__())
            indices      = self.__get_batch(batch)
            metav        = self.__get_metav_by_indeces(indices) if self.metav   is not None else None
            weights      = self.weights[indices]                if self.weights is not None else None
            if self.verbose: print('Retreived metav {}'.format(batch))

        return self.batch(metav = metav, weights = weights)

    def get_dataset(self, **kwargs):
        '''
        Retrieve subset of the data set entailed by self.__indices.
        Be careful, this may be quite memory intensive.
        '''

        # Return just the inputs and outputs.
        return self.__get_data_by_indeces(self.__indices, include_weight = False, include_meta = False, **kwargs)  
        
    def on_epic_end(self):
        '''
        Called after each training epic by Sequence class when being iterated over. 
        '''    
        if self.shuffle: np.random.shuffle(self.__indices)

def plot_wforms(w, x_values = None, figsize = (40,10), xlim = None, ylim = None, ylabels = None, xlabels = 'channels', data_format = 'channels_last'):
    if isinstance(w, tf.Tensor): w = w.numpy()
    N = len(w) if isinstance(w, list) else w.shape[0] if len(w.shape) > 2 else 1
    M = w.shape[-1] if len(w.shape) > 1 else 0 
    fig, axs = plt.subplots(N, M, figsize = figsize)

    def get_ax(i, j):
        if N > 1 and M > 1:
            return axs[i, j]
        elif N > 1:
            return axs[i]
        elif M > 1:
            return axs[j] 
        else:
            return axs

    def get_w(i, j):  
        if isinstance(w, list):
            return w[i][:, j] if data_format == 'channels_last' else w[i][j, :]
        elif N > 1:
            return w[i, :, j] if data_format == 'channels_last' else w[i, j, :]
        elif M > 1:
            return w[i, :, j] if data_format == 'channels_last' else w[i, j, :]
        else:
            return w[:, j]    if data_format == 'channels_last' else w[j, :]

    for i in range(N):
        for j in range(M):
            ax  = get_ax(i, j)
            w_ij = get_w(i, j)
            ax.plot(w_ij) if x_values is None else ax.plot(x_values, w_ij)
            if ylim: ax.set_ylim(*ylim)
            if xlim: ax.set_xlim(*xlim)
    for i in range(N):
        ax = get_ax(i, 0)
        ax.set_ylabel(ylabels[i] if ylabels is not None else 'velocity')
    
    if xlabels == 'channels': xlabels = ['channel {}'.format(j + 1) for j in range(M)]
    if xlabels is not None:
        for j in range(M):
            ax = get_ax(N-1, j)
            ax.set_xlabel(xlabels[j])
         
    return fig, axs

def plot_hists(w, figsize = (40,10), xlabels = None, ylabels = None, xlim = None, ylim = None, data_format = 'channels_last', density = None):
    
    if isinstance(w, tf.Tensor): w = w.numpy()
    N = len(w) if isinstance(w, list) else w.shape[0] if len(w.shape) > 2 else 1
    M = w.shape[-1] if len(w.shape) > 1 else 0 

    fig, axs = plt.subplots(N, 3, figsize = figsize)
    
    def get_ax(i, j):
        if N > 1 and M > 1:
            return axs[i, j]
        elif N > 1:
            return axs[i]
        elif M > 1:
            return axs[j] 
        else:
            return axs

    def get_w(i, j):  
        if isinstance(w, list):
            return w[i][:, j] if data_format == 'channels_last' else w[i][j, :]
        elif N > 1:
            return w[i, :, j] if data_format == 'channels_last' else w[i, j, :]
        elif M > 1:
            return w[i, :, j] if data_format == 'channels_last' else w[i, j, :]
        else:
            return w[:, j]    if data_format == 'channels_last' else w[j, :]
    
    for i in range(N):
        for j in range(M):
            ax    = get_ax(i, j)
            w_ij  = get_w (i, j)
            ax.hist(w_ij, density = density)
            if ylim: ax.set_ylim(*ylim)
            if xlim: ax.set_xlim(*xlim)

    for i in range(N):
        ax = get_ax(i, 0)
        ax.set_ylabel(ylabels[i] if ylabels is not None else 'velocity')

    if xlabels == 'channels': xlabels = ['channel {}'.format(j + 1) for j in range(M)]
    if xlabels is not None:
        for j in range(M):
            ax = get_ax(N-1, j)
            ax.set_xlabel(xlabels[j])

    return fig, axs

def lineplot(lower_line, middle_line, upper_line, loglog = False, xscale = None, yscale = None, xlim = None, ylim = None, x_list = None, ax = None, line_lable = 'line', x_label = 'x', y_label = 'y', title = 'Plot', color = None):
    
    x_list = x_list if x_list is not None else range(len(middle_line))
    
    # Create the plot object
    fig, ax = (None, ax) if ax is not None else plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    kwargs = {}
    if color     : kwargs['color'] = color
    if line_lable: kwargs['label'] = line_lable
    
    # ax.loglog(x_list, middle_line, lw = 1, alpha = 1, **kwargs)
    if not loglog:
        ax.plot(x_list, middle_line, lw = 1, alpha = 1, **kwargs)
    else:
        ax.loglog(x_list, middle_line, lw = 1, alpha = 1, **kwargs)

    # Adjust x,y scaling.
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)

    # Shade the confidence interval
    kwargs = {}
    if color     : kwargs['color'] = color
    ax.fill_between(x_list, lower_line, upper_line, alpha = 0.4, **kwargs)
    
    if ylim: ax.set_ylim(*ylim)
    if xlim: ax.set_xlim(*xlim)
    
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')


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
    w = data.wform[:, :, 0:10]
    print('in  shape:', w.shape      )
    print('out shape x, y:', f(w)[0].shape, f(w)[1].shape)

    weights     = (1 / data.metad['dist']) * np.min(data.metad['dist'])
    SG_train    = SeisGenerator(w, 3, f, metav = np.column_stack([ meta for k, meta in data.metad.items() if k in ['dist', 'magn'] ]), indices = range(10), verbose = True, shuffle = True,  load_data = 'load-and-extract')

    print(SG_train.random_metav().metav.shape)

    x = SG_train.random_batch().x
    print(type(x), x.shape)