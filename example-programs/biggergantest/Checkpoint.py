from tensorflow import train
import os


class Checkpoint:
    def __init__(self, checkpoint_dir,
                 generator, discriminator):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = train.Checkpoint(
            generator_optimizer=generator.optimizer,
            discriminator_optimizer=discriminator.optimizer,
            generator=generator.model,
            discriminator=discriminator.model
        )

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def restore(self, checkpoint_name):
        path = os.path.join(self.checkpoint_dir, checkpoint_name)
        self.checkpoint.restore(path)
