import tensorflow as tf

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2,2), padding='same'),
        #tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(1,1), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2,2), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
        ]

        self.metric_fake_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_real_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
