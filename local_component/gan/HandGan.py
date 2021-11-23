import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

from gan.Discriminator import Discriminator, discriminator_loss
from gan.Generator import generator, generator_loss
from tensorflow.python.client import device_lib
import datetime

log_dir = "logs/"
output_image_path = "../doc/train1/hand1/hand"
input_image_path = "../doc/train1/pose1/pose"
test_output_image_path = "../doc/train1/hand1/test"
test_input_image_path = "../doc/train1/pose1/test"
base_image_path = "../doc/train1/train"
test_base_image_path = "../doc/train1/test"

image_path = "../doc/train2/train"
test_image_path = "../doc/train2/test"

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# summary_writer = tf.summary.FileWriter(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

summary_writer = log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#tf.enable_eager_execution()

#sess = tf.Session(config=tf.ConfigProto(
#      allow_soft_placement=True, log_device_placement=True))

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 150

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 2
    input_image = image[:, :w, :]
    real_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(name):
    input_image, real_image = load(name)
    input_image, real_image = resize(input_image, real_image,
                                     IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))
    '''f = open(summary_writer, "a")
    f.write('epochs ' + str(epoch) + '\n')
    f.write('gen_total_loss' + str(gen_total_loss) + '\n')
    f.write('gen_gan_loss' + str(gen_gan_loss) + '\n')
    f.write('gen_l1_loss' + str(gen_l1_loss) + '\n')
    f.write('disc_loss' + str(disc_loss) + '\n')
    f.close()'''

    '''global summary_writer
    summary_writer.add_summary('gen_total_loss', gen_total_loss)
    summary_writer.add_summary('gen_gan_loss', gen_gan_loss)
    summary_writer.add_summary('gen_l1_loss', gen_l1_loss)
    summary_writer.add_summary('disc_loss', disc_loss)'''


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target)
        print("Epoch: ", epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
    checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    print(f'Hi beautiful {"Ioana"}')
    print(get_available_gpus())

    inp, re = load(image_path + "/0.jpg")

    '''plt.figure()
    plt.imshow(inp)
    plt.figure()
    plt.imshow(re)
    plt.show()'''
    train_dataset = tf.data.Dataset.list_files(image_path + "/*.jpg")
    train_dataset = train_dataset.map(load_image_train)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.list_files(test_image_path + "/*.jpg")
    test_dataset = test_dataset.map(load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    inp, re = resize(inp, re, IMG_HEIGHT, IMG_WIDTH)
    # print(inp.shape)
    # print(re.shape)
    generator = generator()

    gen_output = generator(inp[tf.newaxis, ...], training=False)
    '''plt.imshow(gen_output[0, ...])
    plt.show()'''

    discriminator = Discriminator()
    disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
    '''plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
    plt.colorbar()
    plt.show()'''

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    for example_input, example_target in test_dataset.take(1):
        generate_images(generator, example_input, example_target)

    fit(train_dataset, EPOCHS, test_dataset)
