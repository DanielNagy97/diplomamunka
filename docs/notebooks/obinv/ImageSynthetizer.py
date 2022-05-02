import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class ImageSynthetizer(object):
    def __init__(self, generator, classifier):
        self.generator = generator
        self.classifier = classifier
        self.cross_entropy = keras.losses.CategoricalCrossentropy(
            from_logits=False
        )

    def gradient_descent_momentum(self, goal_label, starting_noise,
                                  step_size=0.005, momentum=0.9, steps=20,
                                  verbose=False):
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