import tensorflow as tf
import matplotlib.pyplot as plt

from obinv.ModelLoader import ModelLoader
from obinv.InputHandler import InputHandler
from obinv.Encoder import Encoder
from obinv.StructuredDataMaker import StructuredDataMaker
from obinv.ImageSynthetizer import ImageSynthetizer
from obinv.SynonymsLoader import SynonymsLoader

class TextToImage(object):
    def __init__(self, generator_path, classifier_path, synonyms_path):
        self.generator = ModelLoader.load_model(generator_path)
        self.classifier = ModelLoader.load_model(classifier_path)
        self.synonyms = SynonymsLoader.load_synonyms(synonyms_path)

        self.input_handler = InputHandler()
        self.structured_data_maker = StructuredDataMaker(self.synonyms)
        self.encoder = Encoder()

        self.image_synthetizer = ImageSynthetizer(self.generator,
                                                  self.classifier)

    def generate_image_from_text(self, input_text,
                                 step_size=0.005, momentum=0.9, steps=20,
                                 verbose=False):
        structured_data = self.structured_data_maker(
                            self.input_handler(input_text),
                            threshold=0.8
                          )
        one_hot_label = self.encoder.make_one_hot(structured_data)

        starting_noise = tf.random.normal([1, 100])

        result_noises, losses, preds =\
            self.image_synthetizer.gradient_descent_momentum(
                one_hot_label,
                starting_noise,
                step_size, momentum, steps, verbose
            )

        self.show_result(result_noises[-1], input_text)

        return result_noises, losses, preds, structured_data

    def show_result(self, result_noise, input_sentence):
        generated_image = self.generator(result_noise, training=False)[0]
        fig_1 = plt.figure(figsize=(3, 3), dpi=100)
        ax = fig_1.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.set_title('"' + input_sentence + '"', y=-0.1)
        ax.imshow(
            (generated_image[0].numpy()*127.5+127.5).astype("uint8"),
            interpolation="none"
        )
        plt.show()

    def plot_convergence(self, result_noises, losses, preds,
                         image_step, dpi=100):
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

        axs['Classes'].plot(preds, label=self.synonyms.keys())

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
            generated_image = self.generator(result_noises[i*image_step],
                                             training=False)[0]
            inserted[i].axis('off')
            inserted[i].set_title(i*image_step)
            inserted[i].imshow(
                (generated_image[0].numpy()*127.5+127.5).astype("uint8"),
                interpolation="none"
            )
        plt.show()