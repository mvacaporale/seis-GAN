#-------------------------------------------
# ParserUtils.py: 
#      Parser classes for model inputs.
#-------------------------------------------

from   argparse import ArgumentParser
import datetime
import time

class ArgModelParser(ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(ArgModelParser, self).__init__(*args, **kwargs)
        self.args     = None # To be filled on parse - filled by ArgumentParser
        self.config_d = {}   # To be filled on parse - filled by this class using self.args
        self.add_arguments()

    def add_arguments(self):
        self.add_argument( '-t' , '--TrainFrac'        , type   = float       , help = "Fraction of data set to use for training - default 0.74"              , default = 0.74)
        self.add_argument( '-e' , '--Epochs'           , type   = int         , help = "Number of training of epochs - default 200."                          , default = 200 )
        self.add_argument( '-d' , '--ProjectDir'       , type   = str         , help = "Name of project directory. default = <today's iso date>"              , default = None)
        self.add_argument( '-r' , '--Restore'          , type   = str         , help = "Whether to restore model and continue training from checkpoint."      , default = False)
        self.add_argument( '-rc', '--RestoreCheckpoint', type   = str         , help = "Name of particular checkpoint to restore."                            , default = None)
        self.add_argument( '-v' , '--Verbose'          , type   = bool        , help = "Whether to print verbose feedback."                                   , default = True)
        self.add_argument( '-n' , '--Notes'            , type   = str         , help = "Notes about the model."                                               , default = None )

    def parse_args(self):
        '''
        Parse arguments and define config_d.
        '''
        self.args = super(ArgModelParser, self).parse_args()

        self.config_d = {
            
            #---------
            # Data
            #---------
            
            'data_path'   : '/seis/wformMat_jpm4p_181106_downsample-5x.h5', # path to data
            'data_format' : 'channels_last', # data format ('channels_first' or 'channels_last')
            'train_frac'  : self.args.TrainFrac,  # % of data devoted to Training
            
            #---------
            # Wforms
            #---------

            'burn_seconds'                 : 2.5,   # first  part of wform to throw away
            'input_seconds'                : 80,    # middle part of waveform to use as input
            'output_seconds'               : None,  # last   part of waveform to use as target output or None if generting x.
            'measure_rate'                 : 20,    # sampling rate in HZ
            'normalize_data'               : True,  # whether to normalize input waveforms to lie between -1 and 1.
            'predict_normalization_factor' : True,  # whether to predict the normalization factor if computed
            
            #---------
            # Training
            #---------
            
            'batch_size'            : 256, # batch size
            'epochs'                : self.args.Epochs,  # training epochs - default 200
            'random_seed'           : 7, # random seed
            'load_data_into_memory' : 'load', # 'load-and-extract',
            'metas'                 : ['dist', 'magn'], # meta to include as conditional parameters.
            'weight_loss'           : False,
            'lr_gen'                : 1e-4,
            'lr_dis'                : 1e-4,
            # 'clip_gen_grad_norm'    : 0.35,

            "conditional_config"    : {"one_hot_encode": True, "aux_classify": False},
            "bins_s"                : [10, [3, 4, 4.5, 5, 5.5, 6, 7, "inf"]],
            "num_channels"          : 3,
            # "transformation_name"   : "arc_sinh",
            # "transformation_paramd" : {"pre_scale" : 0.01},

            #--------
            # Saving
            #--------
            
            'directory'      : self.args.ProjectDir or './GAN_{}'.format(datetime.datetime.now().date().isoformat()), # defaults to todays date. Will make a dir called 'GAN_<directory>
            'restore'        : self.args.Restore or bool(self.args.RestoreCheckpoint),
            'restore_ckpt'   : self.args.RestoreCheckpoint, # Defaults to last checkpoint.
            
            #--------
            # Debug
            #--------

            'debug' : True,

            #--------
            # Misc
            #--------
            'model_notes' : self.args.Notes,
        } 

        # Return args as would the super's function.
        return self.args

if __name__ == '__main__':
    
    # Try out the argument parse.
    parser = ArgModelParser(description = 
        '''
        Run GAN on seismic waveforms. 
        '''
    )

    args = parser.parse_args() 
    if args.Verbose: print('ProjectDir:', args.ProjectDir, args.ProjectDir)
    if args.Verbose: print('Restore   :', args.ProjectDir, args.Restore)

    import json
    print(json.dumps(parser.config_d, indent=4, sort_keys=True))



           
