from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like
from tensorflow import zeros_like


class Discriminator:
    def __init__(self, learning_rate, beta_1, beta_2):
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=learning_rate)
        self.model = self.make_discriminator_model()

    def loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss
        return total_loss

    def add_downsampling_unit(self, model, filters,
                              kernel_size, strides, padding):
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding, activation="relu",
                kernel_initializer="he_normal"
            )
        )
        model.add(BatchNormalization())

    def make_discriminator_model(self):
        model = Sequential()

        model.add(
            Conv2D(
                filters=64,
                kernel_size=4,
                strides=2,
                input_shape=(64, 64, 3),
                padding="same", activation="relu",
                kernel_initializer="he_normal"
            )
        )
        model.add(BatchNormalization())

        self.add_downsampling_unit(model, filters=128,
                                   kernel_size=4, strides=2, padding="same")

        self.add_downsampling_unit(model, filters=256,
                                   kernel_size=4, strides=2, padding="same")

        self.add_downsampling_unit(model, filters=512,
                                   kernel_size=4, strides=2, padding="same")

        model.add(
            Conv2D(
                filters=1,
                kernel_size=4,
                strides=1,
                input_shape=(64, 64, 3),
                padding="valid"
            )
        )

        model.add(Flatten())
        model.add(Activation("sigmoid"))
        return model
