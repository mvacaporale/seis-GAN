import tensorflow as tf
import numpy      as np
import time
import os
import shutil

from   GAN          import GAN
from   ModelBase    import ModelBase

from keras.utils.vis_utils import plot_model

class GANModelBase(ModelBase):
    
    def __init__(self, config):
        
        super(GANModelBase, self).__init__(config)

        # Save GAN file.
        shutil.copyfile('GAN.py', os.path.join(self.project_dir, 'GAN.py'))

    def noise(self):
        # Generate noise from a uniform distribution
        m = 3 # self.config,get('num_channels', 3) @
        return np.random.normal(size = [self.batch_size, m, self.gan.noise_dim]).astype(dtype = np.float32) if self.data_format == 'channel_first' else np.random.normal(size = [self.batch_size, self.gan.noise_dim, m]).astype(dtype = np.float32)

    def build_graph_and_session(self):
        
        def tensor_norms(tensors):
            if isinstance(tensors, list):
                return [tf.norm(g) for g in tensors if g is not None]
            else:
                return tf.map_fn(lambda g: tf.norm(g), tensors)

        def clip_tensors_by_norm(tensors, clip):
            if isinstance(tensors, list):
                return [tf.clip_by_norm(g, clip) for g in tensors if g is not None]
            else:
                return tf.map_fn(lambda g: tf.clip_by_norm(g, clip), tensors)

        self.graph = tf.Graph()
        with self.graph.as_default():
        
            self.gan           = GAN(data_format = self.data_format, verbose = True, aux_classify = self.config.get('aux_classify', False), aux_categories_sizes = self.config.get('aux_categories_sizes', []), predict_normalization_factor = self.config.get('predict_normalization_factor'))
            self.generator     = self.gan.generator
            self.discriminator = self.gan.discriminator


            with tf.device('/device:GPU:0'):

                self.discriminator_optimizer = tf.train.AdamOptimizer(self.config.get('lr_dis', 1e-4), beta1 = 0, beta2 = 0.9)
                self.generator_optimizer     = tf.train.AdamOptimizer(self.config.get('lr_gen', 1e-4), beta1 = 0, beta2 = 0.9)
                self.global_step        = tf.train.get_or_create_global_step()
                
                # Setup conditional placeholder.
                if self.conditional_metav is not None:
                    self.conditional_placeholders = [tf.placeholder(tf.float32, shape = [None, m.shape[1]], name = 'conditional') for m in (self.conditional_metav if isinstance(self.conditional_metav, list) else [self.conditional_metav])]
                    for cp in self.conditional_placeholders:
                        tf.add_to_collection('conditionals', cp)
                else:
                    self.conditional_placeholders = None
                    
                # Setup loss weight tensor.
                if self.config.get('weight_loss', False):
                    self.weights_placeholder = tf.placeholder(tf.float32, shape = [None, 1], name = 'weights') 
                else:
                    self.weights_placeholder = None
                    
                # Set up input and output placeholders.
                self.training_bool_placeholder                            = tf.placeholder_with_default(True, shape = [], name = 'training_bool')
                self.gen_input_placeholder                                = tf.placeholder(tf.float32, shape = [None, self.gan.noise_dim, 3], name = 'latent') # @

                gen_output                                                = self.generator(self.gen_input_placeholder, training = self.training_bool_placeholder, conditional = self.conditional_placeholders)
                self.gen_output_tensor                                    = gen_output.data      # generated data
                self.pred_norm_fake_tensor                                = gen_output.pred_norm # predicted normalization factor (None if predict_normalization_factor = False)
                self.pred_norm_real_placeholder                           = tf.placeholder(tf.float32, shape = [None, 3]) if self.pred_norm_fake_tensor is not None else None
                
                gen_out_training_false                                    = self.generator(self.gen_input_placeholder, training = False, conditional = self.conditional_placeholders).data

                self.discriminator(gen_out_training_false, training = False, conditional = self.conditional_placeholders, pred_norm = self.pred_norm_real_placeholder)

                self.dis_input_placeholder                                = tf.placeholder(tf.float32, shape = self.gen_output_tensor.shape, name = 'real_input')
                
                if hasattr(self.discriminator, 'prep_real_input'):
                    self.dis_input_tensor = self.discriminator.prep_real_input(self.dis_input_placeholder)
                else:
                    self.dis_input_tensor = self.dis_input_placeholder

                dis_output_real                                           = self.discriminator(self.dis_input_tensor , training = self.training_bool_placeholder, conditional = self.conditional_placeholders, pred_norm = self.pred_norm_real_placeholder)
                dis_output_fake                                           = self.discriminator(self.gen_output_tensor, training = self.training_bool_placeholder, conditional = self.conditional_placeholders, pred_norm = self.pred_norm_fake_tensor     )
                self.dis_output_real_tensor                               = dis_output_real.score      # "realness" of real data
                self.dis_ac_loss_real_tensor                              = dis_output_real.extra_loss # extra loss to add if aux_classify = True
                self.dis_output_fake_tensor                               = dis_output_fake.score      # "realness" of fake data
                self.dis_ac_loss_fake_tensor                              = dis_output_fake.extra_loss # extra loss to add if aux_classify = True

                # Name generator tensors.
                with tf.name_scope('gen'):

                    # Name input / output tensor.
                    tf.identity(self.gen_input_placeholder, 'input')
                    tf.identity(self.gen_output_tensor    , 'output')
                    tf.identity(self.pred_norm_fake_tensor, 'pred_norm')

                # Name discriminator tensors.
                with tf.name_scope('dis'):

                    # Name input.
                    tf.identity(self.dis_input_placeholder, 'input')

                    # Name real output tensor.
                    with tf.name_scope('real'):
                        tf.identity(self.dis_output_real_tensor, 'output')     
                    
                    # Name fake output tensor.
                    with tf.name_scope('fake'):
                        tf.identity(self.dis_output_fake_tensor, 'output') 

                #-------------------
                # Generator Step
                #--------------------
                
                # Weight the loss tensor.
                if self.weights_placeholder is not None:
                    dis_output_fake_tensor_weighted = tf.multiply(self.dis_output_fake_tensor, self.weights_placeholder)
                else:
                    dis_output_fake_tensor_weighted = self.dis_output_fake_tensor

                # Define gen loss tensor.
                self.gen_loss_tensor = tf.multiply(tf.constant(-1, dtype = tf.float32), tf.reduce_mean(dis_output_fake_tensor_weighted)) + self.dis_ac_loss_real_tensor + self.dis_ac_loss_fake_tensor
                 
                # Calculate gradients. 
                self.gen_calc_grads_op    = self.generator_optimizer.compute_gradients(self.gen_loss_tensor, self.generator.variables) 
                self.gen_calc_grads_op_2  = tf.gradients(self.gen_loss_tensor, self.generator.variables) 

                # Remove null gradients.
                gen_grads_idx            = [i for i, g_op in enumerate(self.gen_calc_grads_op) if g_op[0] is not None]
                self.generator_variables = [self.gen_calc_grads_op[i][1] for i in gen_grads_idx]
                self.gen_calc_grads_op   = [self.gen_calc_grads_op[i][0] for i in gen_grads_idx]

                # Clip gradients.
                if self.config.get('clip_gen_grad_norm'):
                    self.gen_calc_grads_op = [g_op for g_op in self.gen_calc_grads_op if g_op[0] is not None] if isinstance(self.gen_calc_grads_op, list) else self.gen_calc_grads_op
                    self.gen_calc_grads_op, gen_grads_global_norm = tf.clip_by_global_norm(self.gen_calc_grads_op, self.config['clip_gen_grad_norm'])

                # Apply gradients.
                self.gen_apply_grads_op = self.generator_optimizer.apply_gradients(zip(self.gen_calc_grads_op, self.generator_variables), global_step = self.global_step)

                #-------------------
                # Discriminator Step
                #--------------------

                # Define dis loss tensor.
                epsilon                    = tf.random_uniform([], 0, 1)
                self.mixed_input_data      = (epsilon * self.dis_input_tensor          ) + (1 - epsilon) * self.gen_output_tensor
                self.mixed_input_pred_norm = (epsilon * self.pred_norm_real_placeholder) + (1 - epsilon) * self.pred_norm_fake_tensor if self.pred_norm_fake_tensor is not None else None
                self.mixed_output          = self.discriminator(self.mixed_input_data, training = self.training_bool_placeholder, conditional = self.conditional_placeholders, pred_norm = self.mixed_input_pred_norm).score

                # Compute gradient of mixed_output with respect to it's input - used for regularization.
                mix_calc_grads_op_data       = self.discriminator_optimizer.compute_gradients(self.mixed_output, self.mixed_input_data)
                mix_calc_grads_op_pred_norm  = self.discriminator_optimizer.compute_gradients(self.mixed_output, self.mixed_input_pred_norm) if self.mixed_input_pred_norm is not None else []
                mix_calc_grads_op            = mix_calc_grads_op_data + mix_calc_grads_op_pred_norm

                mix_calc_grads_op  = [g_op[0]                for g_op in  mix_calc_grads_op] # Get just the gradients (in the 0th index) and leave the variables (in the 1st index).
                mix_calc_grads_op  = [tf.reshape(g_op, [-1]) for g_op in  mix_calc_grads_op] # Flatten out the tensors: tf.norm calls tf.reshape(inputs, [-1]) anyhow after concatenating them, but if we do the reverse, it enables different shaped grads_op (which is helpful for when mix_calc_grads_op_pred_norm is not None)
                norm_of_mixed      = tf.norm(tf.concat(mix_calc_grads_op, axis = 0))

                # Weight the logits.           
                if self.weights_placeholder is not None:
                    dis_output_fake_tensor_weighted = tf.multiply(self.dis_output_fake_tensor, self.weights_placeholder)
                    dis_output_real_tensor_weighted = tf.multiply(self.dis_output_real_tensor, self.weights_placeholder)
                else:
                    dis_output_fake_tensor_weighted = self.dis_output_fake_tensor
                    dis_output_real_tensor_weighted = self.dis_output_real_tensor
                                                                
                # Defined loss tensor.
                regularizer = 10 * tf.square(norm_of_mixed - 1)        
                self.dis_loss_tensor = tf.reduce_mean(dis_output_fake_tensor_weighted - dis_output_real_tensor_weighted + regularizer) + 0.25 * self.dis_ac_loss_real_tensor + 0.25 * self.dis_ac_loss_fake_tensor
                
                # Define gradient step tensor. 
                self.dis_calc_grads_op  = self.discriminator_optimizer.compute_gradients(self.dis_loss_tensor, self.discriminator.variables)

                # Remove null gradients.
                dis_grads_idx                = [i for i, g_op in enumerate(self.dis_calc_grads_op) if g_op[0] is not None]
                self.discriminator_variables = [self.dis_calc_grads_op[i][1] for i in dis_grads_idx]
                self.dis_calc_grads_op       = [self.dis_calc_grads_op[i][0] for i in dis_grads_idx]

                # Apply gradients.
                self.dis_apply_grads_op = self.discriminator_optimizer.apply_gradients(zip(self.dis_calc_grads_op, self.discriminator_variables))
                
                #-----------------
                # Diagnostics
                #-----------------

                gen_grads_norm = tf.reduce_mean(tensor_norms(self.gen_calc_grads_op))
                dis_grads_norm = tf.reduce_mean(tensor_norms(self.dis_calc_grads_op))
                mix_grads_norm = tf.reduce_mean(tensor_norms(mix_calc_grads_op))

                real_logits  = tf.reduce_mean(self.dis_output_real_tensor)
                fake_logits  = tf.reduce_mean(self.dis_output_fake_tensor)
                mixed_logits = tf.reduce_mean(self.mixed_output)

            #----------------------
            # Summaries and prints.
            #----------------------
            with tf.name_scope('GradientNorms'):
                # Generator.
                tf.summary.scalar('gen_grads_norm', gen_grads_norm)
                tf.add_to_collection('Prints', tf.Print(gen_grads_norm, [gen_grads_norm], message = ' '*5 + 'gen_grads_norm: '))
               
                # Generator.
                if self.config.get('clip_gen_grad_norm'):
                    tf.summary.scalar('gen_grads_global_norm', gen_grads_global_norm)
                    tf.add_to_collection('Prints', tf.Print(gen_grads_global_norm, [gen_grads_global_norm], message = ' '*5 + 'gen_grads_global_norm: '))

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

            with tf.name_scope('Aux_Classes'):
                # Real
                tf.summary.scalar('real_acc', self.dis_ac_loss_real_tensor)
                tf.add_to_collection('Prints', tf.Print(self.dis_ac_loss_real_tensor, [self.dis_ac_loss_real_tensor], message = ' '*5 + 'self.dis_ac_loss_real_tensor: '))

                # Fake
                tf.summary.scalar('fake_acc', self.dis_ac_loss_fake_tensor)
                tf.add_to_collection('Prints', tf.Print(self.dis_ac_loss_fake_tensor, [self.dis_ac_loss_fake_tensor], message = ' '*5 + 'self.dis_ac_loss_fake_tensor: '))

            # Merge summaries and prints.
            self.summary_op    = tf.summary.merge_all()
            self.init_op       = tf.initializers.global_variables()
            self.print_op      = tf.group(*tf.get_collection('Prints'))
            self.update_op     = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.saver         = tf.train.Saver()

        # Print model descriptions to file.
        summary_file = os.path.join(self.diagnostics_dir, 'model_summary.txt')
        with open(summary_file, 'w') as f:
            f.write('Generator Summary:\n')
            s = self.generator    .summary(print_fn = lambda summary: [f.write(summary + '\n'), print(summary) if self.generator.verbose     else None])
            f.write('\n'*4)

            f.write('Discriminator Summary:\n')
            s = self.discriminator.summary(print_fn = lambda summary: [f.write(summary + '\n'), print(summary) if self.discriminator.verbose else None])
            f.write('\n')

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth = True)
        sessConfig  = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, gpu_options = gpu_options)
        self.sess   = tf.Session(config = sessConfig, graph = self.graph)
        self.sw     = tf.summary.FileWriter(self.diagnostics_dir, self.sess.graph)

        # Initialize global variables or restore checkpoint.
        if self.config.get('restore'): 
            # Restore gloabel variables.
            restore_path = os.path.join(self.checkpoint_dir, self.config.get('restore_ckpt')) if self.config.get('restore_ckpt') else tf.train.latest_checkpoint(self.checkpoint_dir)
            print('Loading the model from checkpoint: %s' % restore_path)
            self.saver.restore(self.sess, restore_path)
        else:
            # Initialize global variables.
            tf.set_random_seed(self.random_seed)
            self.sess.run(self.init_op) 

class GANModel(GANModelBase):
    
    def __init__(self, config):
        super(GANModel, self).__init__(config)
    
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
                    
                    feed_dict = {self.gen_input_placeholder : self.noise(), self.dis_input_placeholder : wform_x}
                    if self.conditional_placeholders is not None:
                        feed_dict.update({cp : m for cp, m in zip(self.conditional_placeholders, batch.metav if isinstance(batch.metav, list) else batch.metav)})
                    if self.weights_placeholder is not None:
                        feed_dict[self.weights_placeholder]        = batch.weights[:, None]
                    if self.pred_norm_real_placeholder is not None:
                        feed_dict[self.pred_norm_real_placeholder] = batch.x_normalization

                    if i % 50 == 1:
                        _, _, summary, step, _ = self.sess.run([self.dis_apply_grads_op, self.print_op, self.summary_op, self.global_step, self.update_op], feed_dict = feed_dict)
                        print('     ... Saving summaries at global step:', step)
                        self.sw.add_summary(summary, step)
                    else:
                        self.sess.run(self.dis_apply_grads_op, feed_dict = feed_dict)

                # Take one learning step for generator.
                batch   = self.SG_train.random_metav()
                
                feed_dict = {self.gen_input_placeholder : self.noise()}
                if self.conditional_placeholders is not None:
                    feed_dict.update({cp : m for cp, m in zip(self.conditional_placeholders, batch.metav if isinstance(batch.metav, list) else batch.metav)})
                if self.weights_placeholder is not None:
                    feed_dict[self.weights_placeholder]     = batch.weights[:, None]

                self.sess.run([self.gen_apply_grads_op, self.update_op], feed_dict = feed_dict)

            # Save the model (checkpoint) every epochs.
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'ganckpt.ckpt'), global_step = self.global_step)
            print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
