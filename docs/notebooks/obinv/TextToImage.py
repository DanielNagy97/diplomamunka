import tensorflow as tf

from obinv.ModelLoader import ModelLoader
from obinv.InputHandler import InputHandler
from obinv.Encoder import Encoder
from obinv.StructuredDataMaker import StructuredDataMaker
from obinv.ImageSynthetizer import ImageSynthetizer
from obinv.SynonymsLoader import SynonymsLoader
from obinv.ResultPlotter import ResultPlotter


class TextToImage(object):
    """
    TextToImage class
    """
    def __init__(self, generator_path, classifier_path, synonyms_path):
        """
        Initializes the class with the given parameters.
        :param generator_path: The path of the generator model.
        :param classifier_path: The path of the classifier model.
        :param synonyms_path: The path of the synonyms json file.
        """
        self.generator = ModelLoader.load_model(generator_path)
        self.classifier = ModelLoader.load_model(classifier_path)
        self.synonyms = SynonymsLoader.load_synonyms(synonyms_path)
        
        self.result_plotter = ResultPlotter()
        self.input_handler = InputHandler()
        self.structured_data_maker = StructuredDataMaker(self.synonyms)
        self.encoder = Encoder()

        self.image_synthetizer = ImageSynthetizer(self.generator,
                                                  self.classifier)

    def generate_image_from_text(self, input_text,
                                 step_size=0.005, momentum=0.9, steps=21,
                                 seed=None, image_step=5, dpi=100,
                                 verbose=False, show_convergence=False):
        """
        Generates image from text.
        :param input_text: The input string.
        :param step_size: The step_size value for the gradient descent.
        :param momentum: The momentum value for the gradient descent.
        :param steps: The steps for the gradient descent search.
        :param seed: The seed for the random point generator.
        :param image_step: The image step in the bottom of the convergence plot.
        :param dpi: The dpi value of the plots.
        :param verbose: To show the generated images in the steps or not.
        :param show_convergence: To plot the convergence figure or not.
        """
        structured_data = self.structured_data_maker(
                            self.input_handler(input_text),
                            threshold=0.8
                          )
        one_hot_label = self.encoder.make_one_hot(structured_data)

        if(seed != None):
            tf.random.set_seed(seed)
        starting_noise = tf.random.normal([1, 100])

        result_noises, losses, preds =\
            self.image_synthetizer.gradient_descent_momentum(
                one_hot_label,
                starting_noise,
                step_size, momentum, steps, verbose
            )

        self.result_plotter.show_result(
            result_noises[-1], input_text, self, dpi
        )

        if(show_convergence):
            self.result_plotter.plot_convergence(
                result_noises, losses, preds, image_step, self, dpi
            )

        return result_noises, losses, preds, structured_data