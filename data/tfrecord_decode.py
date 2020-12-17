import tensorflow as tf
import argparse
import os
import augmentation as aug


class Parser(object):
    """A class to decode data and preprocessing from tfrecord file"""
    def __init__(self, input_size, grid_number, num_class, mode='train'):
        """Create instance for parser

        Args:
            input_size (int): input image size of the neural netowrk
            grid_number (int): denoted S in the paper
            num_class (int): number of output class
            mode (str, optional): oen of 'train', 'val'. Defaults to 'train'.
        """
        self.input_size = input_size
        self.grid_number = grid_number
        self.num_class = num_class
        self._is_training = mode == 'train'
        self.DECODE_FEATURES = {"image": tf.io.FixedLenFeature([], tf.string),
                                "height": tf.io.FixedLenFeature([], tf.int64),
                                "width": tf.io.FixedLenFeature([], tf.int64),
                                "num_obj": tf.io.FixedLenFeature([], tf.int64),
                                "mask": tf.io.VarLenFeature(tf.string),
                                "box": tf.io.VarLenFeature(tf.float32),
                                "id": tf.io.VarLenFeature(tf.int64)}
        if mode == 'train':
            self.augmentations = [aug.flipHorizontal,
                                 aug.randomBrightness,
                                 aug.randomContrast,
                                 aug.randomHUE,
                                 aug.randomSaturation]
        else:
            self.augmentations = []


    def __call__(self, example):
        parsed_tensor = tf.io.parse_single_example(example, features = self.DECODE_FEATURES)
        data = self._decode(parsed_tensor)
        image, cat, mask = self._parse(data)
        return image, cat, mask


    def build_dataset(self, tfrecord_path, batch_size=8, num_epoch=None):
        """Build a dataset object from tfrecord input

        Args:
            tfrecord_path (str): a basepath to tfrecord files
            batch_size (int, optional): batch size. Defaults to 8.
            num_epoch (int, optional): number of epoch to repeat. Only affect in train mode.
            If set to None, the dataset will reapeat indefinitely. Defaults to None.

        Returns:
            tf.data.Dataset: dataset object
        """
        basedir = os.path.dirname(tfrecord_path)
        tfrecord_basename = os.path.basename(tfrecord_path)
        tfrecord_filename, tfrecord_extension = os.path.splitext(tfrecord_basename)
        files = tf.io.matching_files(os.path.join(basedir, tfrecord_filename + '*' + tfrecord_extension))
        num_shards = tf.cast(tf.shape(files)[0], tf.int64)

        dataset = tf.data.Dataset.from_tensor_slices(files)
        if self._is_training:
            if num_epoch is None:
                dataset = dataset.repeat()
            else:
                dataset = dataset.repeat(num_epoch)
        dataset = dataset.shuffle(num_shards, reshuffle_each_iteration=True)
        dataset = dataset.interleave(tf.data.TFRecordDataset,
                                    cycle_length=num_shards,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(2 * batch_size)
        dataset = dataset.map(map_func=self, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


    def _decode(self, parsed_tensor):
        """Decode tfreocrd example back to original data

        Args:
            parsed_tensor (tf.Tensor): a parsed example

        Returns:
            dict: A dictionary with same keys that store decoded/extracted data
        """
        image = tf.io.decode_jpeg(parsed_tensor['image'])
        num_obj = parsed_tensor['num_obj'].numpy()
        height = parsed_tensor['height'].numpy()
        width = parsed_tensor['width'].numpy()

        if num_obj > 0:
            mask_feature = tf.sparse.to_dense(parsed_tensor['mask'])
            mask_feature = tf.map_fn(tf.io.decode_png,
                                     mask_feature,
                                     fn_output_signature=tf.uint8)  # shape (num_obj, H, W, 1)
            mask_feature = tf.squeeze(mask_feature, axis=[3])       # shape (num_obj, H, W)
            mask_feature = tf.cast(mask_feature, tf.float32)

            box_feature = tf.sparse.to_dense(parsed_tensor['box'])
            box_feature = tf.reshape(box_feature, [-1, 4])          # shape (num_obj, 4), format (xmin,ymin,w,h)
            id_feature = tf.sparse.to_dense(parsed_tensor['id'])    # shape (num_obj)
        else:
            mask_feature = tf.zeros([0, height, width], dtype=tf.float32)
            box_feature = tf.zeros([0, 4], dtype=tf.float32)
            id_feature = tf.zeros([0], dtype=tf.int64)

        return {"image": image,
                "num_obj": num_obj,
                "mask": mask_feature,
                "box": box_feature,
                "id": id_feature,
                "width": width,
                "height": height}


    def _parse(self, data):
        """Parse data to match with the network input and label before feeding to the network.
        Also apply augmentation in training mode

        Args:
            data (dict): a dictionary that sotre decoded data. Typically this is
            otuput from _decode method

        Returns:
            tuple: (image, category, mask) tensors
        """
        image = tf.image.resize(data['image'], [self.input_size, self.input_size])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        box_x_min = tf.slice(data['box'], [0, 0], [data['num_obj'], 1])
        box_y_min = tf.slice(data['box'], [0, 1], [data['num_obj'], 1])
        box_width = tf.slice(data['box'], [0, 2], [data['num_obj'], 1])
        box_height = tf.slice(data['box'], [0, 3], [data['num_obj'], 1])
        box_center_x_grid = (box_x_min + box_width / 2) / data['width'] * self.grid_number      # shape (num_obj, 1)
        box_center_y_grid = (box_y_min + box_height / 2) / data['height'] * self.grid_number    # shape (num_obj, 1)
        box_center_x_grid = tf.cast(box_center_x_grid, dtype=tf.int32)
        box_center_y_grid = tf.cast(box_center_y_grid, dtype=tf.int32)
        box_center_grid = tf.stack([box_center_x_grid, box_center_y_grid], axis=1)              # shape (num_obj, 2)
        box_center_grid = tf.squeeze(box_center_grid, axis=2)

        cat = tf.scatter_nd(box_center_grid,
                            data['id'],
                            tf.constant([self.grid_number, self.grid_number]))                  # shape (S, S)
        cat = tf.one_hot(cat, self.num_class, dtype=tf.float32)                                 # shape (S, S, C)

        k = self.grid_number * box_center_y_grid + box_center_x_grid                            # shape (num_obj, 1)
        mask = tf.scatter_nd(k,
                             data['mask'],
                             tf.constant([self.grid_number * self.grid_number, data['height'], data['width']]))    # shape (S^2, H, W)
        mask = tf.transpose(mask, perm=[1,2,0])                                                 # shape (H, W, S^2)
        mask = tf.image.resize(mask, [self.input_size, self.input_size])                        # shape (input_size, input_size, S^2)

        for augment in self.augmentations:
            if tf.random.uniform([1]) > 0.5:
                image, cat, mask = augment(image, cat, mask)

        return image, cat, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode data from tfrecord file(s)')
    parser.add_argument("--tfrecord_path", type=str, required=True,
                    help="Path to tfrecord file(s)")
    args = parser.parse_args()

    parser = Parser(input_size=512, grid_number=24, num_class=91)
    dataset = parser.build_dataset(args.tfrecord_path)

    for image, cat, mask in dataset.take(1):
        print(tf.math.reduce_sum(cat))
