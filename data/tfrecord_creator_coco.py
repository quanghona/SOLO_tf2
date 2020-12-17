import os
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from tfrecord_util import *
import argparse
from tqdm import tqdm
from PIL import Image
import io


def create_tfrecord_coco(annotation_path, image_dir, output_path, num_shards=100):
    """Create TFrecord files from coco dataset (images and annotation)
    Note: The function automatically assign indexes to generated shards.
    And each shard can be use independently.
    Example:
    if output_path is dataset.tfrecord and num_shards = 10 then the output will
    generate a list of files dataset_1.tfrecord, dataset_2.tfrecord, ... dataset_10.tfrecord

    Features format of each example in tfrecord:
     - "image": bytes feature, a jpeg decoded string of the image
     - "height": int64 feature, original height of the image
     - "width": int64 feature, original width of the image
     - "num_obj" int64 feature, number of labeled objects in the images
     - "mask": bytes feature, a list of instance masks, each element in the list
     is png decoded string of the mask image (same size to the image)
     - "box": float list feature, object bounding boxes folowing the coco bounding
     box format, the list is flatten. To decode back, please reshape the list to shape (num_obj, 4)
     - "id": int64 feature, a list of category id of the objects, and have the same order to the box feature

    If num_obj = 0, the mask will be a empty string, box is an empty list and id also an empty list

    Args:
        annotation_path (str): path to coco annotation file (json)
        image_dir (str): directory to images
        output_path (str): the base path to save the tfrecord file
        num_shards (int, optional): Number of shards to split into. Defaults to 100.
    """
    coco = COCO(annotation_path)
    image_ids = coco.getImgIds()

    output_dir = os.path.dirname(output_path)
    output_basename = os.path.basename(output_path)
    output_filename, output_extension = os.path.splitext(output_basename)
    tfrecord_paths = [os.path.join(output_dir, output_filename + '_' + str(i + 1) + output_extension) for i in range(num_shards)]
    tfrecord_writers = [tf.io.TFRecordWriter(file_name) for file_name in tfrecord_paths]

    for i,image_id in tqdm(enumerate(image_ids), total=len(image_ids), desc=annotation_path):
        img_obj = coco.loadImgs(image_id)[0]
        annIds = coco.getAnnIds(image_id)
        annotations = coco.loadAnns(annIds)

        num_obj = len(annotations)
        img_width = img_obj['width']
        img_height = img_obj['height']
        image_path = os.path.join(image_dir, img_obj['file_name'])

        with open(image_path, 'rb') as f:
            image_string = f.read()

        features = {
            "image": bytes_feature(image_string),
            "height": int64_feature(img_height),
            "width": int64_feature(img_width),
            "num_obj": int64_feature(num_obj)
        }
        if num_obj != 0:
            masks = []
            boxes = []
            ids = []
            for annotation in annotations:
                boxes.append(annotation["bbox"])
                ids.append(annotation["category_id"])
                mask = coco.annToMask(annotation)
                pil_image = Image.fromarray(mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                masks.append(output_io.getvalue())

            boxes = np.stack(boxes, axis=0)    # shape (num_obj, 4)
            ids = np.stack(ids, axis=0)        # shape (num_obj)

            # Flatten all matrices before feeding to feature dictionary
            features["mask"] = bytes_list_feature(masks)
            features["box"] = float_list_feature(boxes.reshape(-1))
            features["id"] = int64_list_feature(ids.reshape(-1))
        else:
            features["mask"] = bytes_list_feature("")
            features["box"] = float_list_feature([])
            features["id"] = int64_list_feature([])

        example = tf.train.Example(features=tf.train.Features(feature=features))
        tfrecord_writers[i % num_shards].write(example.SerializeToString())

    for i in range(num_shards):
        tfrecord_writers[i].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create tfrecord for coco dataset')
    parser.add_argument("--annotation", type=str, required=True,
                        help="Path to coco annotation file (json)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory to images")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save tfrecord file(s)")
    parser.add_argument("--num_shards", type=int, default=100,
                        help="Number of shards to split to")
    args = parser.parse_args()

    create_tfrecord_coco(args.annotation, args.image_dir, args.output_path, args.num_shards)
