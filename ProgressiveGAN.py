import tensorflow as tf
from   Layers import *

import inspect

class ProgressiveGANError(Exception):
    pass

class ProgressiveGAN(type):

    # Define scopes to create progressive tensors. 
    progressive_generator_scope     = 'progressive_generator'
    progressive_discriminator_scope = 'progressive_discriminator'
    tracking_tensors_name_scope     = 'progressive_tracking_tensors'

    # Create helper tensors stored by the class - to be defined when graph is built.
    initial_steps_placeholder       = None
    stabilizing_steps_placeholder   = None
    stage_int_tensor                = None
    fade_phase_bool                 = None 
    alpha_tensor                    = None

    def __new__(cls, name, bases, attrs):
        """
        Wrap the call methods of the class, either a Generator or Discriminator.
        This assumes the call method returns (for the Gen.) or receives (for the Dis.)
        the lowest level resolution of the samples. 

        TODO:
        Specify needs for 
            - ToSample
            - FromSample
            - self.blocks
            - self.resizings
        """
        if 'call' in attrs:

            # Turn into progressive call for a Generator.
            if 'gen' in name.lower():
                attrs['call'] = cls.progressive_generator(attrs['call'])

            # Turn into progressive call for a Discriminator.
            elif 'dis' in name.lower():
                attrs['call']            = cls.progressive_discriminator(attrs['call'])

                # Add function to prep real input - resize input according to the current stage.
                attrs['prep_real_input'] = cls.progressive_real_input()

            # Assert the class must be one of the two.
            else:
                raise ProgressiveGANError("Class must be a Discriminator or Generator.")

            #TODO validate attrs

        return super(ProgressiveGAN, cls).__new__(cls, name, bases, attrs)

    #------------
    # Utilities
    #------------

    # TODO Add validations.
    # @classmethod
    # def validate_common_attrs(cls, attrs):
    #     pass

    # @classmethod
    # def validate_dis_attrs(cls, attrs);

    #     cls.validate_common_attrs(attrs)

    # @classmethod
    # def validate_gen_attrs(cls, attrs);

    #     cls.validate_common_attrs(attrs)

    @classmethod
    def apply_layers(cls, blocks, x, **kwargs):
        """
        Recursively apply list of blocks to x.
        Done in order as in b[n](...b[1](b[0](x))).
        """

        # Validate blocks.
        assert isinstance(blocks, list), "Blocks must be supplied as a list."
        assert len(blocks) >= 1        , "Must supply non-empty list of blocks."

        # Base Case: Apply remaining block to x.
        if len(blocks) == 1:
            return blocks[0](x, **kwargs)

        # Recursive Case: Apply first block to x and then apply the remaining ones.
        else:
            x = blocks[0](x)
            return cls.apply_layers(blocks[1:], x, **kwargs)

    @classmethod
    def get_training_param_from_args(cls, call, args, kwargs):
        """
        Evaluate args (list) and kwargs (dict) to see if 
        training has been 
        """

        # Inspect arguments of call.
        argspec = inspect.getargspec(call) 
        
        # Determine if 'training' is specified in kwargs.
        if 'training' in kwargs:
            training = kwargs['training']

        # Determine if 'training' is specified in args.
        elif 'training' in argspec.args:

            # Determine index of training arg.
            training_arg_idx = argspec.args.index('training')

            # Get training arg is present among those passed.
            if len(args) > training_arg_idx:
                training_passed = args[training_arg_idx] 
            else:
                training_passed = None

            # Determine index of training arguments among defaults.
            training_default_idx = training_arg_idx - len(argspec.defaults)

            # Get default training argument if present among defaults.
            if training_default_idx < len(argspec.defaults):
                training_default = argspec.defaults[training_default_idx]
            else:
                training_default = None

            # Set training to be of the arg specified or it's default or, lastly, None.
            training = training_passed or training_default
        
        # Default training None.
        else:
            training = None

        # Return training boolean.
        return training

    @classmethod
    def get_or_create_tracking_tensors(cls):
        """
        Get or create helper tensors to keep track of progression.
        """

        # Get default graph.
        default_graph = tf.get_default_graph()

        # Get current tracking tensors - these may be all be None. 
        tracking_tensors = [cls.initial_steps_placeholder, cls.stabilizing_steps_placeholder, cls.stage_int_tensor, cls.fade_phase_bool, cls.alpha_tensor]

        # Redefine tracking tensors if any are None or belong to a different graph then the default.
        if None in tracking_tensors or False in [t.graph == default_graph for t in tracking_tensors]:

            # Go to the top-level (empty) name scope.
            with tf.name_scope(None):
                
                # Embody tracking tensors in new name_scope.
                with tf.name_scope(cls.tracking_tensors_name_scope):
                    
                    # Get the global step.
                    global_step_var = tf.train.get_or_create_global_step()

                    # Get the number of steps desired for the initial training of phase 0 of the foundational blocks.
                    cls.initial_steps_placeholder     = tf.placeholder_with_default(tf.constant(10000, tf.int64), shape = [], name = 'stabilizing_steps') 

                    # Get the number of steps desired for stabilizing phase or default to 10,000. 
                    # We'll assume this equals the number of steps for the fade-in phase as well.
                    cls.stabilizing_steps_placeholder = tf.placeholder_with_default(tf.constant(10000, tf.int64), shape = [], name = 'stabilizing_steps') 

                    # Define tensor to keep track of the current stage of progression. 
                    # For instance, if there are 4 resizings, the stage_int_tensor will take on values 0 through 4. 
                    stage_float_tensor = 1 + tf.divide(                  # add one to start the count after phase 0
                        global_step_var - cls.initial_steps_placeholder, # subtract to account for initial steps of phase 0
                        2 * cls.stabilizing_steps_placeholder            # mul. by 2 to account for fade-in and stabilizing phases 
                    )
                    cls.stage_int_tensor = tf.cond(
                        global_step_var < cls.initial_steps_placeholder, # check global steps have surpassed initial steps of phase 0
                        lambda: tf.constant(0, dtype = tf.int32), 
                        lambda: tf.cast(tf.floor(stage_float_tensor), tf.int32) 
                    )

                    # Define tensor to identify whether or not we're in the fading-in phase.
                    fade_phase_float = stage_float_tensor - tf.cast(cls.stage_int_tensor, stage_float_tensor.dtype)
                    cls.fade_phase_bool = tf.cond(
                        global_step_var < cls.initial_steps_placeholder, # check global steps have surpassed initial steps of phase 0
                        lambda: tf.constant(False), 
                        lambda: fade_phase_float < 0.5, 
                    )

                    # Define tensor to keep track of the step within the current stage of progression. 
                    stabilizing_step_tensor = tf.floormod(global_step_var, cls.stabilizing_steps_placeholder) 

                    # Define tensor to keep track of fractional (0 to 1) development within the current stage of progression.
                    alpha_tensor     = tf.divide(stabilizing_step_tensor, cls.stabilizing_steps_placeholder)
                    cls.alpha_tensor = tf.cast(alpha_tensor, tf.float32)

                    # Return newly made tracking tensors.
                    return [cls.initial_steps_placeholder, cls.stabilizing_steps_placeholder, cls.stage_int_tensor, cls.fade_phase_bool, cls.alpha_tensor]

        # Otherwise return them as there are.
        else:
            return tracking_tensors

    #-----------------------------------
    # Progressive tensor-maker functions
    #-----------------------------------

    @classmethod
    def build_progressive_output(cls, N_stages, get_x_stabilize, get_x_fade, get_final_progression):
        """
        Function to build progressive output tensor - one that
        conditionally returns the output through the appropriate
        blocks of the network given the current stage and whether or
        not we're fading-in a block.

        Args:
        N_stages              - (int)      number of total stages - enumeration assumed to start at zero
        get_x_stabilize       - (callable) function that takes one arg, 'stage', and returns the corresponding stabilizing tensor
        get_x_fade            - (callable) function that takes one arg, 'stage', and returns the corresponding fading tensor
        get_final_progression - (callable) function that takes no  args and returns a the final progression - a pass through all blocks

        Return: tf.Tensor - specifically one from tf.case. 

        Note: All aforementioned callables must give the same type of output - see the documentation of tf.cast for specifics.
        """

        #-----------------------------------------------------
        # Define function to help build progressive output.
        #-----------------------------------------------------

        # Get tensors to help keep track of the progression
        initial_steps_placeholder, stabilizing_steps_placeholder, stage_int_tensor, fade_phase_bool, alpha_tensor = cls.get_or_create_tracking_tensors()

        def populate_progressive_pred_fn_list(pred_fn_list, stage, fade_phase):
            '''
            This function populates a given list with predicate-function pairs 
            that sequentially build up the blocks of the network 
            following 'stage' and 'fade_phase' inclusively. 
            The populated list may be used to construct a conditional tensor with tf.case.

            For instance, with args:

                pred_fn_list = [], stage = 0, fade_phase = False

            one would have after running this function:

                pred_fn_list = [
                    (predicate of 0th   stage when stabilizing, pass through first block)
                    (predicate of 1st   stage when fading-in  , pass through first block + pass through fading-in second block)
                    ...
                    (predicate of final stage when stabilizing, pass through all blocks)
                ]

            Args:
            pred_fn_list - (list) list to populate (starts empty and gets mutated)
            stage        - (int)  stage of progression
            fade_phase   - (bool) whether or not to build the fade-in phase

            Return: None
            '''

            # Define predicate of conditional statements.
            predicate = (stage_int_tensor <= stage) & tf.equal(fade_phase_bool, tf.constant(fade_phase))

            # Base Case: Final stage and after the last fade-in.
            if stage == N_stages and not fade_phase:
                
                # Append pass through all blocks - to occur when all blocks except the last (i.e. 'call') have been stabilized.
                pred_fn_list.append((predicate, get_final_progression))

            # Other Cases: Previous stages.
            elif not fade_phase:

                # Append pass through all blocks up to the current one to be stabilized.
                pred_fn_list.append((predicate, lambda: get_x_stabilize(stage)))

                # Populate with remaining progressions.
                populate_progressive_pred_fn_list(pred_fn_list, stage + 1, fade_phase = True)

            else:
                
                # Append pass through fading-in an fading-out blocks.
                pred_fn_list.append((predicate, lambda: get_x_fade(stage)))

                # Populate with remaining progressions.
                populate_progressive_pred_fn_list(pred_fn_list, stage, fade_phase = False)

        #-----------------------------------------------------
        # Build and return progression.
        #-----------------------------------------------------

        # Populate progressive_pred_fn_list - a list of predicate-function pairs that sequentially build the up the blocks of the network.
        progressive_pred_fn_list = []
        populate_progressive_pred_fn_list(progressive_pred_fn_list, stage = 0, fade_phase = False) # Populate list.

        # Build output progressive tensor - the final case (pass through all blocks) is set as the default.
        output_progression_tensor = tf.case(pred_fn_pairs = progressive_pred_fn_list[:-1], default = progressive_pred_fn_list[-1][1])

        # Return tf.case tensor.
        return output_progression_tensor

    @classmethod
    def progressive_generator(cls, call):
        """
        Return a wrapper to 'call' that transforms the output to the progressive form.
        That is, one where the output of the Generator grows in the size of the 
        resolution given the current global step. 
        """
        def build_progressive_blocks(self, *args, **kwargs):
            
            #---------------------------
            # Prep progression.
            #---------------------------

            # Apply first block to input.
            args          = list(args)
            x             = args[0]
            output_tuple  = call(self, *([x] + args[1:]), **kwargs) # Yields tuple.
            x_block_0     = output_tuple[0]
            
            # Get training param.
            training = cls.get_training_param_from_args(call, args, kwargs)
 
            # Define the first block to return the first part of output_tuple.
            def first_block(x, training = training):
                return x_block_0 # Lowest-res sample.

            # Get block, resizing, and from_sample layers.
            blocks    = [first_block] + list(self.blocks) # Note: tf makes self.blocks into a ListWrapper, hence the added typecast to ensure order is maintained.
            resizings = self.resizings
            to_samps  = self.to_samps

            # Get tensors to help keep track of the progression
            initial_steps_placeholder, stabilizing_steps_placeholder, stage_int_tensor, fade_phase_bool, alpha_tensor = cls.get_or_create_tracking_tensors()

            #-----------------------------------------------
            # Define helper functions to build progression.
            #-----------------------------------------------

            # Define function to pass input through all blocks.
            def get_final_progression():

                # Apply remaining blocks.
                x_from_blocks = cls.apply_layers(blocks, x) # Yields tensor.
                x_to_samp     = self.to_samps[-1](x_from_blocks)

                # Incorporate the yielded highest-res sample into output_tuple and return. 
                return self.gen_output(x_to_samp, *output_tuple[1:])

            # Return pass through all blocks when in training mode. 
            if training is False:
                return get_final_progression() # Highest-res sample.
            
            # Define function to fade-in (new) layers and fade-out (old) layers.
            def get_x_fade(stage):

                x_fade_out = resizings[stage - 1](to_samps[stage - 1](
                        cls.apply_layers(
                            blocks[:stage], 
                            x,
                            training = training
                        )
                ))
                x_fade_in = to_samps[stage](
                        cls.apply_layers(
                            blocks[:(stage + 1)], 
                            x,
                            training = training
                        )
                )

                # Parameterize the fade by alpha for each tensor and construct new output tuple.
                x_fade = (1 - alpha_tensor) * x_fade_out + alpha_tensor * x_fade_in
                
                # Incorporate other output tensors from first block (i.e. call).
                x_fade = self.gen_output(x_fade, *output_tuple[1:])
                return x_fade

            # Define function to stabilize layer at a given stage. 
            def get_x_stabilize(stage):

                x_stabilize = to_samps[stage](
                        cls.apply_layers(
                            blocks[:(stage + 1)], 
                            x,
                            training = training
                        )
                )
            
                # Incorporate other output tensors from first block (i.e. call).
                x_stabilize = self.gen_output(x_stabilize, *output_tuple[1:])
                return x_stabilize

            #-----------------------------------------------------
            # Build progressive output.
            #-----------------------------------------------------

            # Embody progressive generator in new scope.
            with tf.name_scope(cls.progressive_generator_scope):

                # Build output tuple of progressive tensors - it's a tuple as 'call' returns a tuple.
                output_progressive_tuple = cls.build_progressive_output(len(blocks) - 1, get_x_stabilize, get_x_fade, get_final_progression) 

                # Conditionally return non-progressive output when the param 'training' is a passed as a tensor.
                if isinstance(training, tf.Tensor):
                    output_progressive_tuple = tf.cond(training, lambda: output_progressive_tuple, get_final_progression)

            # Return tuple of progressive outputs.
            return output_progressive_tuple

        # Return function to build progressive blocks.
        return build_progressive_blocks

    @classmethod
    def progressive_discriminator(cls, call):
        """
        Return a wrapper to 'call' that transforms the output to the progressive form.
        That is, one where the output of the Discriminator grows in the size of the 
        resolution given the current global step. 
        """
        def build_progressive_blocks(self, *args, **kwargs):
            
            #---------------------------
            # Prep progression.
            #---------------------------

            # Apply blocks and from_samp to input - assumed to be the first part of args.
            args          = list(args)
            x             = args[0]
            
            # Get training param.
            training = cls.get_training_param_from_args(call, args, kwargs)
            
            # Define call to be the last block.
            def last_block(x, training = training):
                return call(self, *([x] + args[1:]), **kwargs)

            # Get block, resizing, and from_sample layers.
            blocks     = list(self.blocks) + [last_block]
            resizings  = self.resizings
            from_samps = self.from_samps

            # Get tensors to help keep track of the progression
            initial_steps_placeholder, stabilizing_steps_placeholder, stage_int_tensor, fade_phase_bool, alpha_tensor = cls.get_or_create_tracking_tensors()

            #-----------------------------------------------
            # Define helper functions to build progression.
            #-----------------------------------------------

            # Define run through all blocks.
            def get_final_progression():
                x_from_samp   = from_samps[0](x)
                x_from_blocks = cls.apply_layers(self.blocks, x_from_samp) 
                return last_block(x_from_blocks)               

            # Run input through all blocks when in training mode.
            if training is False:
                return get_final_progression()

            # Define function to stabilize layer at a given stage.    
            def get_x_stabilize(stage):
                x_stabilize = cls.apply_layers(
                    blocks    [(-1 - stage):],
                    from_samps[ -1 - stage  ](x),
                    training = training
                )
                return x_stabilize

            # Define function to fade-in (new) layers and fade-out (old) layers.
            def get_x_fade(stage):

                x_fade_out = cls.apply_layers( 
                        blocks       [(-1 - (stage - 1)):],
                        from_samps   [ -1 - (stage - 1)  ](
                            resizings[ -1 - (stage - 1)  ](x)
                        ),
                        training = training
                )
                x_fade_in = cls.apply_layers(
                        blocks    [(-1 - stage):],
                        from_samps[ -1 - stage  ](x),
                        training = training
                )

                # Parameterize the fade by alpha for each tensor and construct new output tuple.
                x_fade = self.dis_output(*[(1 - alpha_tensor) * x_fo + alpha_tensor * x_fi for x_fo, x_fi in zip(x_fade_out, x_fade_in)])
                return x_fade

            #-----------------------------------------------------
            # Build and return progression.
            #-----------------------------------------------------

            # Embody progressive discriminator in new scope.
            with tf.name_scope(cls.progressive_discriminator_scope):
                
                # Build output tuple of progressive tensors - it's a tuple as 'call' returns a tuple.
                output_progressive_tuple = cls.build_progressive_output(len(blocks) - 1, get_x_stabilize, get_x_fade, get_final_progression) 
                
                # Conditionally return non-progressive output when the param 'training' is a passed as a tensor (possibly a tf.placeholder).
                if isinstance(training, tf.Tensor):
                    output_progressive_tuple = tf.cond(training, lambda: output_progressive_tuple, get_final_progression)


            # Return tuple of progressive outputs.
            return output_progressive_tuple

        # Return function to build progressive blocks.
        return build_progressive_blocks

    @classmethod
    def progressive_real_input(cls):
        """
        Return a wrapper to that transforms the input to the progressive form.
        That is, one where the input is resized to match the current stage of the
        Discriminator. 
        """

        #------------------------------------------------
        # Build progressive input tensor with tf.case.
        #------------------------------------------------
        def build_progressive_real_input(self, x):

            # Get tensors to help keep track of the progression
            initial_steps_placeholder, stabilizing_steps_placeholder, stage_int_tensor, fade_phase_bool, alpha_tensor = cls.get_or_create_tracking_tensors()
                
            # Get resizing layers and blocks of Discriminator.
            resizings = self.resizings
            blocks    = self.blocks

            def populate_progressive_pred_fn_list(pred_fn_list, stage):
                '''
                This function populates a given list with predicate-function pairs 
                that sequentially builds the input progression 
                following 'stage' and 'fade_phase' inclusively. 
                The populated list may be used to construct a conditional tensor with tf.case.

                For instance, with args:

                    pred_fn_list = [], stage = 0e

                one would have after running this function:

                    pred_fn_list = [
                        (predicate of 0th   stage input, input resized to     smallest resolution)
                        (predicate of 1st   stage input, input resized to 2nd smallest resolution)
                        ...
                        (predicate of final stage input, input kept at same size - no change)
                    ]

                Args:
                pred_fn_list - (list) list to populate (starts empty and gets mutated)
                stage        - (int)  stage of progression

                Return: None
                '''
                
                # Define predicate of conditional statements.
                predicate = (stage_int_tensor <= stage)

                # Base Case: Final Stage - return input as is.
                if stage == len(blocks):

                    # Append input as is without any resizing.
                    pred_fn_list.append((predicate, lambda: x))

                # Other Cases: Previous Stages - resize input for the stage accordingly.
                else:

                    # Append with input resized for the current stage.
                    pred_fn_list.append((predicate, lambda: cls.apply_layers(resizings[stage:], x)))

                    # Populate with remaining input progressions.
                    populate_progressive_pred_fn_list(pred_fn_list, stage + 1)

            #-----------------------------------------------------
            # Build and return progression.
            #-----------------------------------------------------

            # Populate progressive_pred_fn_list - a list of predicate-function pairs that sequentially build the up the blocks of the network.
            progressive_pred_fn_list = []
            populate_progressive_pred_fn_list(progressive_pred_fn_list, stage = 0) # Populate list.

            # Build input progressive tensor - the final case (the non-altered input) is set as the default.
            input_progression_tensor = tf.case(pred_fn_pairs = progressive_pred_fn_list[:-1], default = progressive_pred_fn_list[-1][1])

            # Return tf.case tensor.
            return input_progression_tensor

        # Return function to build progressive input.
        return build_progressive_real_input

if __name__ == '__main__':
    pass