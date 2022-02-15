import time
from tensorflow import function as tf_function
from tensorflow import random
from tensorflow import GradientTape
from catgan.Generator import Generator
from catgan.Discriminator import Discriminator
from catgan.FigurePlotter import FigurePlotter
from catgan.Checkpoint import Checkpoint


class GanModel:
    def __init__(self, latent_dim,
                 batch_size, number_of_examples,
                 learning_rate, beta_1, beta_2):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.generator = Generator(self.latent_dim, learning_rate,
                                   beta_1, beta_2)
        self.discriminator = Discriminator(learning_rate, beta_1, beta_2)
        self.seed = random.normal([number_of_examples, latent_dim])
        self.checkpoint = Checkpoint(
            './training_checkpoints',
            self.generator,
            self.discriminator
        )

    @tf_function
    def train_step(self, images):
        noise = random.normal([self.batch_size, self.latent_dim])

        with GradientTape() as gen_tape, GradientTape() as disc_tape:
            generated_images = self.generator.model(noise, training=True)

            real_output = self.discriminator.model(images,
                                                   training=True)
            fake_output = self.discriminator.model(generated_images,
                                                   training=True)

            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss,
            self.generator.model.trainable_variables
        )

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss,
            self.discriminator.model.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(gradients_of_generator,
                self.generator.model.trainable_variables)
            )
        self.discriminator.optimizer.apply_gradients(
            zip(gradients_of_discriminator,
                self.discriminator.model.trainable_variables)
            )

        return (gen_loss, disc_loss)

    def train(self, dataset, epochs):
        figurePlotter = FigurePlotter()
        for epoch in range(epochs):
            start = time.time()

            for (batch, image_batch) in enumerate(dataset):
                gen_loss, disc_loss = self.train_step(image_batch)

                if batch % 100 == 0:
                    average_batch_loss =\
                        gen_loss.numpy()/int(image_batch.shape[1])
                    print(f"""Epoch {epoch+1}
                            Batch {batch} Loss {average_batch_loss:.4f}""")

            # Saving the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save()

            print('Time for epoch {} is {} sec'.format(epoch + 1,
                                                       time.time()-start))

            example_images = self.generator.model(self.seed, training=False)
            figurePlotter.plot_grid_of_images(example_images, epoch)

        # Generating after the final epoch
        example_images = self.generator.model(self.seed, training=False)
        figurePlotter.plot_grid_of_images(example_images, epochs)
