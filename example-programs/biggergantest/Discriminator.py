from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like
from tensorflow import zeros_like


class Discriminator:
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999):
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
        self.model = self.make_discriminator_model()

    def loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(zeros_like(fake_output), fake_output)

        total_loss = real_loss + fake_loss
        return total_loss

    def add_downsampling_unit(self, model, units):
        model.add(
            Conv2D(filters=units[0], kernel_size=(3, 3),
                padding="same", activation="leaky_relu")
        )
        model.add(
            Conv2D(filters=units[1], kernel_size=(3, 3),
                padding="same", activation="leaky_relu")
        )
        model.add(AveragePooling2D())

    def make_discriminator_model(self):
        model = Sequential()
        
        # out: 64x64x256
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3),
                input_shape=(128, 128, 3), padding="same",
                activation="leaky_relu")
        )
        model.add(
            Conv2D(filters=256, kernel_size=(3, 3),
                padding="same", activation="leaky_relu")
        )
        model.add(AveragePooling2D())
        
        # out: 32x32x512
        self.add_downsampling_unit(model, (256, 512))
        
        # out: 16x16x512
        self.add_downsampling_unit(model, (512, 512))
        
        # out: 8x8x512
        self.add_downsampling_unit(model, (512, 512))
        
        # out: 4x4x512
        self.add_downsampling_unit(model, (512, 512))
        
        # out: last
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3),
                padding="same", activation="leaky_relu")
        )
        model.add(
            Conv2D(filters=512, kernel_size=(4, 4), strides=4,
                padding="same", activation="leaky_relu")
        )
        
        model.add(Flatten())
        model.add(Activation("sigmoid"))

        return model