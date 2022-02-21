from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like


class Generator:
    def __init__(self, latent_dim, learning_rate, beta_1=0.9, beta_2=0.999):
        self.latent_dim = latent_dim
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
        )
        self.model = self.make_generator_model(latent_dim)

    def loss(self, fake_output):
        return self.cross_entropy(ones_like(fake_output), fake_output)

    def add_upsampling_unit(self, model, units):
        model.add(UpSampling2D())
        model.add(
            Conv2D(filters=units[0], kernel_size=(3, 3),
                padding='same')
        )
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(
            Conv2D(filters=units[1], kernel_size=(3, 3),
                padding='same')
        )
        model.add(BatchNormalization())
        model.add(LeakyReLU())

    def make_generator_model(self, latent_dim):
        model = Sequential()
        
        # out: 4x4x512
        model.add(Reshape((1, 1, latent_dim), input_shape=[latent_dim]))
        model.add(
            Conv2DTranspose(filters=512, kernel_size=(4, 4),
                            strides=(2, 2), padding="valid")
        )
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(
            Conv2D(filters=512, kernel_size=(3, 3),
                padding='same')
        )
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        assert model.output_shape == (None, 4, 4, 512)
        
        # out: 8x8x512
        self.add_upsampling_unit(model, (512, 512))
        assert model.output_shape == (None, 8, 8, 512)
        
        # out: 16x16x512
        self.add_upsampling_unit(model, (512, 512))
        assert model.output_shape == (None, 16, 16, 512)
        
        # out: 32x32x512
        self.add_upsampling_unit(model, (512, 512))
        assert model.output_shape == (None, 32, 32, 512)
        
        # out: 64x64x256
        self.add_upsampling_unit(model, (256, 256))
        assert model.output_shape == (None, 64, 64, 256)
        
        # out: 128x128x128
        self.add_upsampling_unit(model, (128, 128))
        assert model.output_shape == (None, 128, 128, 128)
        

        model.add(Conv2D(3, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        return model