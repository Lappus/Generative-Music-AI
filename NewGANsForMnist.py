import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import os 
import tensorboard

(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()  

#print("Number of training files:", len(train_images))
#print("Number of testing files:", len(test_images))

# Print the first few files in each set (optional)
#print("Train files:", train_images[:5])
#print("Test files:", test_images[:5])

idx = train_labels==3
filtered_train_images = train_images[idx]
filtered_train_labels = train_labels[idx]

BATCH_SIZE=256
BUFFER_SIZE=60000

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

train_ds = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

filtered_train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
filtered_train_images = (train_images - 127.5) / 127.5

filtered_train_ds = tf.data.Dataset.from_tensor_slices(filtered_train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#---------------------------------------------- Model initialisation ----------------------------------------------#

class Discriminator_model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_model, self).__init__()

        self.convlayer1 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2,2), padding='same', activation='relu', input_shape=[28, 28, 1])
        self.leakyRelu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.convlayer2 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2,2), padding='same', activation='relu')
        self.leakyRelu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.pooling3 = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(1)

    @tf.function(reduce_retracing=True)
    def call(self,x):
        x = self.convlayer1(x)
        x = self.leakyRelu1(x)
        x = self.dropout1(x)
        x = self.pooling1(x)

        x = self.convlayer2(x)
        x = self.leakyRelu2(x)
        x = self.dropout2(x)
        x = self.pooling2(x)

        x = self.pooling3(x)
        x = self.out(x)
        return x 

#class gen_Layer(tf.keras.layers.Layer):
#    def __init__ (self, num_filters):
#        super(gen_Layer, self).__init__()

#       self.convtrans1 = tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=5, strides=(1,1), padding='same')
#        self.convtrans2 = tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=5, strides=(2,2), padding='same')

#    @tf.function(reduce_retracing=True)
#    def call(self, x):
#        x = self.convtrans1(x)
#        x = self.convtrans2(x)
#        return x    

class Generator_model(tf.keras.Model):
    def __init__(self):
        super(Generator_model, self).__init__()

        self.denselayer1 = tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
        self.batchNormalization1 = tf.keras.layers.BatchNormalization()
        self.leakyRelu1 = tf.keras.layers.LeakyReLU()
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))

        self.conv2dTrans1 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1,1), padding='same')
        self.batchNormalization2 = tf.keras.layers.BatchNormalization()
        self.leakyRelu2 = tf.keras.layers.LeakyReLU()

        self.conv2dTrans2 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2,2), padding='same')
        self.batchNormalization3 = tf.keras.layers.BatchNormalization()
        self.leakyRelu3 = tf.keras.layers.LeakyReLU()

        #self.conv2dTrans3 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2,2), padding='same')
        #self.batchNormalization4 = tf.keras.layers.BatchNormalization()
        #self.leakyRelu4 = tf.keras.layers.LeakyReLU()

        self.conv2dTrans4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=(2,2), activation='tanh')

    @tf.function(reduce_retracing=True)
    def call(self, x):
        x = self.denselayer1(x)
        x = self.batchNormalization1(x)
        x = self.leakyRelu1(x)
        x = self.reshape(x)
        x = self.conv2dTrans1(x)
        x = self.batchNormalization2(x)
        x = self.leakyRelu2(x)
        x = self.conv2dTrans2(x)
        x = self.batchNormalization3(x)
        x = self.leakyRelu3(x)
        #x = self.conv2dTrans3(x)
        #x = self.batchNormalization4(x)
        #x = self.leakyRelu4(x)
        x = self.conv2dTrans4(x)
        
        return x 

discriminator = Discriminator_model()
generator = Generator_model()

#---------------------------------------------- Loss and Optimiser ----------------------------------------------#

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # ones_like for the function we want to minimise the distance to
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # zero_like for the function we want to maximise the distance to
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#---------------------------------------------- Checkpoint Saving ----------------------------------------------#

checkpoint_dir = './training_checkpoints/MusicGANs'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)


#---------------------------------------------- Training the model ----------------------------------------------#

EPOCHS = 20
noise_dim = 100
num_examples_to_generate = 9

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function(reduce_retracing=True)
def train_discriminator(images, step):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

    with train_summary_writer.as_default():
        tf.summary.scalar('discriminator_loss', disc_loss, step=step)  


@tf.function(reduce_retracing=True)
def train_generator(step):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)

    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))

    with train_summary_writer.as_default():
        tf.summary.scalar('discriminator_loss', gen_loss, step=step)

train_log_dir = 'logs/train'
test_log_dir = 'logs/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def train(dataset, epochs):
    epochs_parts = epochs/2
    total_batches = len(dataset)
    

    for stage in range(2):
        print('Starting to train stage:', stage+1)
        for epoch_part in range(int(epochs_parts)):
            start_time = time.time()
            image_progress = 0

            for images_batch in dataset:
                train_discriminator(images_batch, image_progress)
                image_progress += 1
                completion_percentage = image_progress/total_batches
            
                if image_progress % 20 == 0 or completion_percentage == 1:
                    print('Training Discriminator: {:.2f}% complete'.format(completion_percentage*100))

            print('Epoch {} took {:.2f} sec'.format(epoch_part+1, time.time()-start_time))

            if (epoch_part + 1) % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                print('Time for epoch {} is {:.2f} sec'.format(epoch_part+1, time.time()-start_time))

        for epoch_part in range(int(epochs_parts)):
            start_time = time.time() 
            image_progress = 0

            for images_batch in dataset:
                train_generator(image_progress)
                image_progress += 1
                completion_percentage = image_progress/total_batches

                if image_progress % 20 == 0 or completion_percentage == 1:
                    print('Training Generator: {:.2f}% complete'.format(completion_percentage*100))

            print('Epoch {} took {:.2f} sec'.format(epoch_part+1, time.time()-start_time))

            epoch = (epoch_part+1)*(stage+1)
            if (epoch_part + 1) % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
                print('Time for epoch {} is {} sec'.format(epoch, time.time()-start_time))
            
            generate_and_save_images(generator, stage+1, epoch, seed)
            #discriminator_loss.reset_states()
            #generator_loss.reset_states()
    generate_and_save_images(generator,stage=3, epoch=epochs, test_input=seed)


'''
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        counter = 0
        total = len(dataset)

        for image_batch in dataset:
            counter+=1
            train_step(image_batch)
            print('Training Model: {} from {} complete in {:.2f} sec'.format(counter, total, time.time()-start))
        generate_and_save_images(generator, stage=1, epoch=epoch+1, test_input=seed)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time for epoch {} is {} sec'.format(epoch+1, time.time()-start))
    
    generate_and_save_images(generator, stage=1, epoch=epochs, test_input=seed)
'''

def generate_and_save_images(model, stage, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('Mnist/image_stage_{:04d}_epoch_{:04d}.png'.format(stage, epoch))


subset_size = 20  
train_subset = filtered_train_ds.take(subset_size)

train(filtered_train_ds, EPOCHS)
