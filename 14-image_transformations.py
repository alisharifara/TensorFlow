# this program performs some image transformations like: resize, flip, and crop, etc.

import tensorflow as tf
from PIL import Image

class ImageTransformation:

    # reads the original images from the directory
    list_of_images = ['./images/cat1.jpg',
                      './images/cat2.jpg',
                      './images/cat3.jpg',
                      './images/cat4.jpg']

    # it creates a queue
    file_name_queue = tf.train.string_input_producer(list_of_images)

    # reads all the images
    image_reader = tf.WholeFileReader()

    with tf.Session() as sess:

        # coordinates loading of the images
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        new_image_list = []

        for i in range(len(list_of_images)):

            # ignore the first value which it returns (it's the file name) and the second one is the actual image file
            _, image_file = image_reader.read(file_name_queue)

            # decode the image as JPEG file.. this converts the image to tensor that we need for training
            image = tf.image.decode_jpeg(image_file)

            image = tf.image.resize_images(image, [150, 150])

            # w:100, h:100, and 3 channels
            image.set_shape((150, 150, 3))

            image = tf.image.central_crop(image, central_fraction=0.4)

            image = tf.image.flip_left_right(image)

            image_array = sess.run(image)
            print("new shape of the image: {}".format(image_array.shape))

            # tf.stack, stacks a list of rank-R tensors into one rank-(R+1) tensor
            image_tensor = tf.stack(image_array)

            print(image_tensor)
            new_image_list.append(image_tensor)

        # it stops and ends the file name coordinator
        coord.request_stop()
        coord.join(threads)

        # tf.stack converts list of images into single tensor with 4 dimensions
        # the output is :Tensor("stack_4:0", shape=(4, 120, 120, 3), dtype=float32)
        # The first dimension indicates the number of images in the list
        image_tensor = tf.stack(new_image_list)
        print(image_tensor)

        writer = tf.summary.FileWriter('./output_3', graph=sess.graph)
        summary = sess.run(tf.summary.image('images', image_tensor, max_outputs=5))
        writer.add_summary(summary)

        writer.close()
