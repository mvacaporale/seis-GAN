#------------------------------------------------------------------------------------------
# run_gan.py:
#       - Instantiate GANModel object via parsed arguments and train.
#------------------------------------------------------------------------------------------

import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['TF_CPP_VMODULE'] = "auto_mixed_precision=2"

import tensorflow as tf

from GANModel     import GANModel
from ParserUtils  import ArgModelParser

# Parse arguments.
parser = ArgModelParser(description = 
    '''
    Run GAN on seismic waveforms. 
    '''
)
args     = parser.parse_args() # <- populates config_d
config_d = parser.config_d

# Print tensorflow info.
from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if args.Verbose: print(get_available_gpus())


if __name__ == '__main__':  

    import time

    start = time.time()
    gm    = GANModel(config_d)
    gm.train()

    print('DONE: total elapsed time:', time.time() - start)