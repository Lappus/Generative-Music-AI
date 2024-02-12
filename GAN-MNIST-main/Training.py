import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime
import matplotlib.pyplot as plt

from GAN import *

NUM_EPOCHS = 50
BATCH_SIZE = 256

def main():

    #
    # Load dataset
    #   

    train_ds = tfds.load("mnist", split="train+test", as_supervised=True)

    train_ds = train_ds.apply(prepare_data)


    #
    # Logging
    #

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Initialize model.
    #

    gan = GAN()

    # Build 
    gan.discriminator.build(input_shape=(None, 28, 28, 1))
    gan.generator.build(input_shape=(1, gan.generator.noise_dim))
    
    # Get overview of number of parameters
    gan.discriminator.summary()
    gan.generator.summary()

    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for img_real in tqdm.tqdm(train_ds, position=0, leave=True): 
            gan.train_step(img_real)

        log(train_summary_writer, gan, epoch)
        generate_and_save_images(GAN.generator, epoch, test_input=seed)
        # Save model (its parameters)
        gan.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")


def log(train_summary_writer, gan, epoch):

    generator_loss = gan.generator.metric_loss.result()
    discriminator_loss = gan.discriminator.metric_loss.result()
    discriminator_fake_loss = gan.discriminator.metric_fake_loss.result()
    discriminator_real_loss = gan.discriminator.metric_real_loss.result()

    gan.generator.metric_loss.reset_states()

    gan.discriminator.metric_loss.reset_states()
    gan.discriminator.metric_fake_loss.reset_states()
    gan.discriminator.metric_real_loss.reset_states()

    #
    # Generate images
    #

    num_generated_imgs = 16
    noise = tf.random.normal(shape=(num_generated_imgs, gan.generator.noise_dim))
    generated_imgs = gan.generator(noise)
    generated_imgs = tf.reshape(generated_imgs, shape=(num_generated_imgs, 28,28,1))
    
    #
    # Write to TensorBoard
    #

    with train_summary_writer.as_default():
        tf.summary.scalar(f"generator_loss", generator_loss, step=epoch)

        tf.summary.scalar(f"discriminator_loss", discriminator_loss, step=epoch)
        tf.summary.scalar(f"discriminator_fake_loss", discriminator_fake_loss, step=epoch)
        tf.summary.scalar(f"discriminator_real_loss", discriminator_real_loss, step=epoch)

        tf.summary.image(name="generated_imgs",data = generated_imgs, step=epoch, max_outputs=num_generated_imgs)
        

    #
    # Output
    #
    print(f"         generator_loss: {generator_loss}")
    print(f"     discriminator_loss: {discriminator_loss}")
    print(f"discriminator_fake_loss: {discriminator_fake_loss}")
    print(f"discriminator_real_loss: {discriminator_real_loss}")
 
 
def prepare_data(dataset):

    dataset = dataset.filter(lambda img, label: label == 0) # only '0' digits

    # Remove label
    dataset = dataset.map(lambda img, label: img)

    # Flatten
    #dataset = dataset.map(lambda img: tf.reshape(img, (-1,)))

    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img: tf.cast(img, tf.float32) )

    #Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img: (img/128.)-1. )

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

seed = tf.random.uniform([16, 100], minval=-1, maxval=1)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('Generative-Music-AI/Mnist/epoch_{:04d}.png'.format(epoch))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")