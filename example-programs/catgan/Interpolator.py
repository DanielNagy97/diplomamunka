import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tensorflow import random


class Interpolator:
    def __init__(self):
        pass

    def interpolate_between_points(self,
                                   generator,
                                   number_of_points):
        if not os.path.isdir("renders"):
            os.mkdir("renders")

        noises = random.normal([number_of_points, generator.latent_dim])
        noises = np.concatenate(
            (
                noises.numpy(),
                noises[0].numpy().reshape(1, generator.latent_dim)
            )
        )

        step_size = 30
        count = 0
        for i in range(1, len(noises)):
            linfit = interp1d(
                [1, step_size], np.vstack([noises[i-1], noises[i]]), axis=0
            )

            for j in range(1, step_size):
                generated_image = generator.model(
                    linfit(j).reshape(1, generator.latent_dim), training=False
                )
                plt.imshow(
                    (generated_image[0].numpy() * 127.5 + 127.5)
                    .astype("uint32")
                )
                count += 1
                plt.savefig('renders/interpol_{:04d}.png'.format(count))
