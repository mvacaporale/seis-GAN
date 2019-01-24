# Import modules.
import os
import matplotlib.pyplot as mp

# Import tools for keras.
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from librosa.display import waveplot
from IPython import display

import datetime
import time

from SeisUtils import extract_func
from SeisUtils import SeisData
from SeisUtils import SeisGenerator
from GAN       import GAN

tf.enable_eager_execution()

# Create project directories.
d = datetime.datetime.now().date().isoformat()
project_dir    = './GAN_{}'.format(d)
checkpoint_dir = os.path.join(project_dir, './checkpoints')
images_dir     = os.path.join(project_dir, './images'     )

if not os.path.exists(project_dir):
    os.makedirs(project_dir)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)    
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

# Import data.
data        = SeisData(data_path = '/seis/wformMat_jpm4p_181106.h5')
# data        = SeisData(data_path = 'wformMat_jpm4p_181106.h5')
f           = data.f            # h5 dictionary
meta        = data.meta         # meta info (13, 260764) includes 'dist' and 'magn'
wform_names = data.wform_names  # names (e.g '/proj..japan...446.UD')
wform       = data.wform        # wave forms - (3, 10500, 260764)
metad       = data.metad        # { 'dist' : [...], 'magn' : [...] } (distance and magnitude dict for easier access)

# Fix random seed for reproducibility
np.random.seed(7)

# Define training, testing, and validation indeces.
N                  = wform.shape[2]
n1, n2             = int(np.ceil(0.80 * N)), int(np.ceil(0.16 * N))
all_indeces        = np.random.choice(N, N, replace = False)
testIdx, validIdx, trainIdx = np.split(all_indeces, [n1, n1 + n2])
testIdx.sort(), validIdx.sort(), trainIdx.sort();
print('Testing    Samples:', len(testIdx ))
print('Validation Samples:', len(validIdx))
print('Training   Samples:', len(trainIdx))

gan           = GAN(data_format = 'channels_first') 
generator     = gan.generator
discriminator = gan.discriminator

# Defun gives 10 secs/epoch performance boost
generator.call     = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)

# Define data generator.
BATCH_SIZE  = 256
weights     = (1 / metad['dist']) * np.min(metad['dist'])
extract_f   = extract_func(data_format = gan.data_format, burn_seconds = 5, pred_seconds = 10)
SG_test     = SeisGenerator(wform, BATCH_SIZE, extract_f, indeces = testIdx , shuffle = False, expend = True)
SG_valid    = SeisGenerator(wform, BATCH_SIZE, extract_f, indeces = validIdx, shuffle = True)
SG_train    = SeisGenerator(wform, BATCH_SIZE, extract_f, indeces = trainIdx, shuffle = True)
print('Testing    Samples:', len(SG_test ))
print('Validation Samples:', len(SG_valid))
print('Training   Samples:', len(SG_train))

def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
generator_optimizer     = tf.train.AdamOptimizer(1e-4)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint        = tf.train.Checkpoint(generator_optimizer     = generator_optimizer,
                                        discriminator_optimizer = discriminator_optimizer,
                                        generator               = generator,
                                        discriminator           = discriminator)

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
num_examples_to_generate = 4
random_vector_for_generation = tf.random_normal([num_examples_to_generate, 3, gan.noise_dim]) if gan.data_format == 'channels_first' else tf.random_normal([num_examples_to_generate, gan.noise_dim, 3]) 

# Create conditional form:
mi_1  = np.argmin(metad['magn'])
mi_2  = np.argmax(metad['magn'])
di_1  = np.argmin(metad['dist'])
di_2  = np.argmax(metad['dist'])
idxs  =  [mi_1, mi_2, di_1, di_2]
idxs.sort()
condx = tf.convert_to_tensor(extract_f(wform[:, :, idxs])[0])

# Concatenate both of them. 
random_vector_for_generation = tf.concat([condx, random_vector_for_generation], 2 if gan.data_format == 'channels_first' else 1)

def generate_and_save_wforms(model, epoch, test_input):
    # print('generate_and_save_wforms')
    # print('test_input.shape', test_input.shape)
    
    predictions = model(test_input, training = False) # training set to False to avoid training batchnorm layer during inference
    # print('predictions.shape', predictions.shape)
    fig = plt.figure(figsize=(4,3))
    for i in range(4):
        for j in range(3):
            plt.subplot(4, 3, i*3 + j + 1)
            plt.plot(predictions[i, :, j].numpy())
    for i in range(4):
        plt.subplot(4, 3, i*3 + 1)
        plt.ylabel('pred {}'.format(i))
    for j in range(3):
        plt.subplot(4, 3, 3*3 + j + 1)
        plt.xlabel('channel {}'.format(j + 1))
    path = os.path.join(images_dir, 'image_at_epoch_{:04d}.png'.format(epoch))
    plt.savefig(path)
    plt.show()   
    
def train(dataset, epochs):  
    data_format = gan.data_format
    cat_dim     = 2 if data_format == 'channels_first' else 1
    noise_dim   = gan.noise_dim
    # print('cat_dim', cat_dim)
    for epoch in range(epochs):
        start = time.time()
        for wform_x, wform_y in dataset:

            wform_x = tf.convert_to_tensor(wform_x)
            wform_y = tf.convert_to_tensor(wform_y)
            
            # meta    = np.tile(meta, (3, 1, 1))
            # meta    = meta.transpose((1, 0, 2)) if data_format == 'channels_first' else meta.transpose((1, 2, 0))
            # meta    = tf.convert_to_tensor(meta)
            
            # print('wform_x.shape', wform_x.shape)
            # print('wform_y.shape', wform_y.shape)
            # print('meta   .shape', meta   .shape)

            # generating noise from a uniform distribution
            noise = tf.random_normal([BATCH_SIZE, 3, noise_dim]) if data_format == 'channels_first' else tf.random_normal([BATCH_SIZE, noise_dim, 3])
            # print('noise.shape', noise.shape)

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_wforms = generator(tf.concat([wform_x, noise], cat_dim), training = True)
                # print('generated_wforms.shape', generated_wforms.shape)
                # print('cat shape', (tf.concat([wform_x, wform_y         ], cat_dim)).shape)
                real_output      = discriminator(tf.concat([wform_x, wform_y         ], cat_dim), training = True)
                generated_output = discriminator(tf.concat([wform_x, generated_wforms], cat_dim), training = True)

                gen_loss  = generator_loss(generated_output)
                disc_loss = discriminator_loss(real_output, generated_output)

            gradients_of_generator     = gen_tape.gradient(gen_loss, generator.variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

            generator_optimizer    .apply_gradients(zip(gradients_of_generator    , generator.variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))


        if epoch % 1 == 0:
            display.clear_output(wait = True)
            generate_and_save_wforms(generator,
                                   epoch + 1,
                                   random_vector_for_generation)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            # print('saving')
            checkpoint.save(file_prefix = checkpoint_prefix + '_' + str(epoch) )

        # print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        # generating after the final epoch
        display.clear_output(wait = True)
        generate_and_save_wforms(generator,
                                 epochs,
                                 random_vector_for_generation)                                 


if __name__ == '__main__':

	EPOCHS      = 10000
	BUFFER_SIZE = 60000
	t1 = time.time()
	train(SG_train, EPOCHS)
	t2 = time.time() - t1
	print('total elapsed time:', t2)