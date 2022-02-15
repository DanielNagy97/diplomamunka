from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like


class Generator:
    def __init__(self, latent_dim, learning_rate, beta_1, beta_2):
        self.latent_dim = latent_dim
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model = self.make_generator_model(latent_dim)

    def loss(self, fake_output):
        return self.cross_entropy(ones_like(fake_output), fake_output)

    def add_upsampling_unit(self, model,
                            filters, kernel_size, strides, padding):
        model.add(
            Conv2DTranspose(
                filters=filters, kernel_size=kernel_size,
                strides=strides,
                padding=padding, activation="relu",
                kernel_initializer="he_normal"
            )
        )
        model.add(BatchNormalization())

    def make_generator_model(self, latent_dim):
        model = Sequential()
        model.add(Reshape((1, 1, 100), input_shape=[latent_dim]))

        self.add_upsampling_unit(model, 512, 4, (1, 1), 'valid')

        self.add_upsampling_unit(model, 256, 4, (2, 2), 'same')

        self.add_upsampling_unit(model, 128, 4, (2, 2), 'same')

        self.add_upsampling_unit(model, 64, 4, (2, 2), 'same')

        model.add(Conv2DTranspose(filters=3, kernel_size=4,
                                  strides=(2, 2), padding='same'))
        model.add(Activation("tanh"))

        return model
