import tensorflow as tf
from PIL import Image

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

        image = tf.image.resize_images(image, [120, 120])

        # w:100, h:100, and 3 channels
        image.set_shape((120, 120, 3))

        image_array = sess.run(image)
        print("new shape of the image: {}".format(image_array.shape))

        Image.fromarray(image_array.astype('uint8'), 'RGB').show()

        # expand_dims, converts a 3-d image tensor into 4-d image tensor
        new_image_list.append(tf.expand_dims(image_array, 0))

    # it stops and ends the file name coordinator
    coord.request_stop()
    coord.join(threads)

    writer = tf.summary.FileWriter('./output', graph=sess.graph)

    i = 0
    for image_tensor in new_image_list:
        summary = sess.run(tf.summary.image("cat_image_" + str(i), image_tensor))
        writer.add_summary(summary)
        i += 1

    writer.close()






