import matplotlib.pyplot as plt


class ResultPlotter(object):
    """
    ResultPlotter class
    """
    def __init__(self):
        pass

    def show_result(self, result_noise, input_sentence, text_to_image,
                    dpi=100):
        """
        Plotting the result image from the noise with the generator.
        :param result_noise: The noise found in the searching process.
        :param input_sentence: The original sentence.
        :param text_to_image: A refference for a TextToImage object.
        :param dpi: The dpi value of the plot.
        """
        generated_image = text_to_image.generator(result_noise,
                                                  training=False)[0]
        fig_1 = plt.figure(figsize=(3, 3), dpi=dpi)
        ax = fig_1.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_title('"' + input_sentence + '"', y=-0.1)
        ax.imshow(
            (generated_image[0].numpy()*127.5+127.5).astype("uint8"),
            interpolation="none"
        )
        plt.show()

    def plot_convergence(self, result_noises, losses, preds,
                         image_step, text_to_image, dpi=100):
        """
        Plotting the convergence from the result of the image generation.
        :param result_noises: The list of noises found during the searching.
        :param losses: The list of losses happend during the searching.
        :param preds: The list of predictions happend during the searching.
        :param image_step: The step between the images.
        :param text_to_image: A refference for a TextToImage object.
        :param dpi: The dpi value of the plot.
        """
        fig = plt.figure(figsize=(8, 4), dpi=dpi, constrained_layout=True)
        axs = fig.subplot_mosaic([['Losses', 'Classes'],['Images', 'Images']],
                                gridspec_kw={
                                    'width_ratios':[2, 2],
                                    'height_ratios':[2, 1]
                                })
        axs['Losses'].set_xlabel('Lépés')
        axs['Losses'].set_ylabel('Távolság')
        axs['Losses'].set_title('A loss változása az iterációkban')
        axs['Losses'].grid(True, color='0.6', dashes=(5, 2, 1, 2))
        axs['Losses'].plot(losses)
        axs['Losses'].set_ylim(bottom=0)

        axs['Classes'].set_ylabel('Valószínűség')
        axs['Classes'].set_xlabel('Lépés')
        axs['Classes'].set_title('Címkék valószínűségi értéke')

        axs['Classes'].grid(True, color='0.6', dashes=(5, 2, 1, 2))

        axs['Classes'].plot(preds, label=text_to_image.synonyms.keys())

        axs['Classes'].set_ylim(bottom=0)
        axs['Classes'].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        inserted = []
        axs['Images'].axis('off')
        pad = 0.1
        for i in range(5):
            inserted.append(
                axs['Images'].inset_axes([(i/10) + pad*i, 0.05, 0.2, 0.9])
            )

        for i in range(5):
            generated_image = text_to_image.generator(
                result_noises[i*image_step],
                training=False)[0]
            inserted[i].axis('off')
            inserted[i].set_title(i*image_step)
            inserted[i].imshow(
                (generated_image[0].numpy()*127.5+127.5).astype("uint8"),
                interpolation="none"
            )
        plt.show()