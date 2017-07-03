import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'orientation 0: 0deg, 1: 90deg, 2: 180deg, 3: 270deg',
}

def get_split(split_name, dataset_dir):
    keys_to_features = {
        'image/encode': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/label': tf.FixedLenFeature([1], tf.int64) 
    }
    
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encode', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/label')
    }
    
    reader = tf.TFRecordReader
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
            data_sources=os.path.join(dataset_dir, 'orient_train_*.tfrecord'),
            reader=reader,
            decoder=decoder,
            num_samples=166*4,
            items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
            num_classes=4)