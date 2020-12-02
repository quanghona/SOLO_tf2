import tensorflow as tf


def preprocess(img_path, annotations, grid_number, num_class):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_image(img_raw)
    img_shape = img_tensor.shape

    # TODO: preprocessing stuff
    cat = tf.zeros((grid_number, grid_number, num_class), dtype=tf.float32)
    mask = tf.zeros((img_shape[0], img_shape[1], grid_number*grid_number), dtype=tf.float32)

    for annotation in annotations:
        seg = annotation["segmentation"]
        box = annotation["bbox"]

        center_x = box[0] + box[2] / 2
        center_y = box[1] + box[3] / 2

        cat[center_y, center_x, annotation["category_id"]] = 1.
        # TODO: map mask to mask label
