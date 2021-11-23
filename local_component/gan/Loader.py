import os
import time
import tensorflow as tf
from matplotlib import pyplot as plt


from gan.Discriminator import Discriminator, discriminator_loss
from gan.Generator import generator, generator_loss
from tensorflow.python.client import device_lib
import datetime

from gan.HandGan import load_image_test, BATCH_SIZE

log_dir = "logs/"
output_image_path = "../doc/train1/hand1/hand"
input_image_path = "../doc/train1/pose1/pose"
test_output_image_path = "../doc/train1/hand1/test"
test_input_image_path = "../doc/train1/pose1/test"
base_image_path = "../doc/train1/train"
test_base_image_path = "../doc/train1/test"

image_path = "../doc/train2/train"
test_image_path = "../dataset/train"
import os

class Loader():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with tf.device("/cpu"):
            self.generator = generator()
            self.discriminator = Discriminator()

            self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
            print("intra in loader")
            checkpoint_dir = './gan/training_checkpoints'
            #checkpoint_dir = './training_checkpoints'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                             discriminator_optimizer=self.discriminator_optimizer,
                                             generator=self.generator,
                                             discriminator=self.discriminator)
            self.status = self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            print("ar trebui sa faca restore")

    def generate_images(self, test_input, tar):
        #self.status.assert_consumed()
        #with tf.device("/cpu"):

        prediction = self.generator(test_input, training=True)
        return prediction[0]
        '''plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()'''







if __name__ == '__main__':
    print(f'Hi beautiful {"Ioana"}')
    loader = Loader()
    test_dataset = tf.data.Dataset.list_files(test_image_path + "/*.jpg")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Run the trained model on a few examples from the test dataset
    with tf.device("/cpu"):
        for inp, tar in test_dataset.take(1):
            print(inp.shape)
            print(tar.shape)
            predictions = loader.generate_images(inp, tar)