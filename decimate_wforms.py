import h5py
import obspy

down_sample = 5 # Factor of downsampling.
data_path_1 = '/seis/wformMat_jpm4p_181106.h5'
data_path_2 = data_path_1.split('.h5')[0] + '_downsample-{}x'.format(down_sample) + '.h5'

print('Opening files: ...')
print('   data_path_1 (to read ):', data_path_1)
print('   data_path_2 (to write):', data_path_2)
print()

f1 = h5py.File(data_path_1, 'r')
f2 = h5py.File(data_path_2, 'w')

print('Copying everything except \'wforms\' for now...') 
for dset in ['numMeta', 'wformFullNames']:
    
    # Create corresponding dataset in f2.
    f2.create_dataset(dset, f1[dset].shape, dtype = f1[dset].dtype)
    
    # Copy over values.
    f2[dset][:] = f1[dset][:]
    
    # Copy attributes. 
    for key, value in  f1[dset].attrs.items():
        f2[dset].attrs[key] = value
print('     done copying numMeta and wformFullNames.\n')

print('Copying wforms...')
wform_len = int(f1["wforms"].shape[1] / down_sample)
f2.create_dataset("wforms", (3, wform_len, 260764), chunks = (3, wform_len, 256))

# Copy attributes. 
for key, value in  f1["wforms"].attrs.items():
    f2["wforms"].attrs[key] = value
    
print('   Attributes copied. Moving on to data ...')
    
for i in range(f1['wforms'].shape[0]):
    for j in range(f1['wforms'].shape[2]):
        
        # Print feedback every 1000.
        if j % 1000 == 0:
            print('   copying and decimating wform[{}, :, {}]'.format(i, j))
        
        # Make, copy, and decimate trace.
        tr     = obspy.core.Trace(data = f1['wforms'][i, :, j])
        tr_new = tr.copy()
        tr_new .detrend()
        tr_new .decimate(factor = 5)
        f2['wforms'][i, :, j] = tr_new.data
print()
       
# Finalize.
print('And that\'s all of them.')
f2.flush()
f2.close()
f1.close()