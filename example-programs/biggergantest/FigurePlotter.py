import os
import matplotlib.pyplot as plt


class FigurePlotter:
    def __init__(self):
        if not os.path.isdir("epochs"):
            os.mkdir("epochs")

    def plot_grid_of_images(self, images, epoch):
        plt.figure(figsize=(8, 8))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow((images[i].numpy() * 127.5 + 127.5).astype("uint32"))
            plt.axis('off')

        plt.savefig('epochs/image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
