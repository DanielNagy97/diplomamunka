import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


class ImageSynthetizer(object):
    """
    ImageSynthetizer class
    """
    def __init__(self, generator, classifier):
        """
        Initializes the synthetizer class.
        :param generator: The trained generator keras.model object.
        :param classifier: The trained classifier keras.model object.
        """
        self.generator = generator
        self.classifier = classifier
        self.cross_entropy = keras.losses.CategoricalCrossentropy(
            from_logits=False
        )

    def gradient_descent_momentum(self, goal_label, starting_noise,
                                  step_size=0.005, momentum=0.9, steps=20,
                                  verbose=False):
        """
        Searching for the given class in the generator's latent space.
        :param goal_label: The one-hot label to maximize.
        :param starting_noise: A random noise for the generator.
        :param step_size: The step size for the gradient descent.
        :param momentum: The momentum for the gradient descent.
        :param steps: The number of steps in the gradient descent.
        :param verbose: To show the generated images in the steps or not.
        :return: (result_noises, losses, preds) lists
        """
        noise = tf.Variable(starting_noise, name='noise')
        
        result_noises = []
        losses = []
        preds = []
        
        change = 0
        for i in range(steps):
            with tf.GradientTape() as g_tape:
                g_tape.watch(noise)

                generated_image = self.generator(noise, training=False)[0]
                
                predictions = self.classifier(generated_image)
                
                loss = self.cross_entropy(goal_label, predictions[0])

            result_noises.append(noise)
            preds.append(predictions[0])
            losses.append(loss)

            gradients = g_tape.gradient(loss, noise)
            change = (step_size * gradients) + momentum * change
            noise = noise - change

            if(verbose):
                print(predictions)
                print(f"Step: {i}, Loss: {loss}")
                plt.imshow(
                    (generated_image[0].numpy()*127.5+127.5).astype("uint8"),
                    interpolation="none"
                )
                plt.show()

        return result_noises, losses, preds