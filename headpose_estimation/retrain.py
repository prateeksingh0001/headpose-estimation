from typing import List

import argparse
import csv
import hashlib
import os
from pathlib import Path
import random
import re
import sys
import tarfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from six.moves import urllib
from tensorflow.contrib.metrics import streaming_pearson_correlation
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

#debugging
from tensorflow.python import debug as tf_debug

VAL = 2 ** 27 + 1
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image retraining for headpose estimation using mutiple FC layers")

    parser.add_argument('--architecture',
                        type=str,
                        dest="architecture",
                        help="architecture to be used in the frontend")
    parser.add_argument('--model_dir',
                        type=str,
                        dest="model_dir",
                        help="The directory where the model is to be saved")
    parser.add_argument('--image_dir',
                        type=str,
                        dest="image_dir",
                        help="The directory to store images")
    parser.add_argument('--labels_file',
                        type=str,
                        dest='labels_file',
                        help='The path to the file with all the labels')
    parser.add_argument('--summaries_dir',
                        type=str,
                        dest="summaries_dir",
                        help="Directory for saving the summaries")
    parser.add_argument('--how_many_training_steps',
                        type=int,
                        dest="how_many_training_steps",
                        help="Num training steps for training the model")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        dest="learning_rate",
                        help="Learning rate for the algo")
    parser.add_argument('--validation_percentage',
                        type=int,
                        dest="validation_percentage",
                        help="percentage of the dataset that is to be used for validation set")
    parser.add_argument('--testing_percentage',
                        type=int,
                        dest="testing_percentage",
                        help="percentage of dataset that is to be used as test set")
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=5,
                        dest="train_batch_size",
                        help="Batch size for training")
    parser.add_argument('--validation_batch_size',
                        type=int,
                        default=5,
                        dest="validation_batch_size",
                        help="Batch size for validation")
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=5,
                        dest="test_batch_size",
                        help="Batch size for testing")
    parser.add_argument('--bottleneck_dir',
                        type=str,
                        dest="bottleneck_dir",
                        help="Directory where the bottleneck tensors are stored")
    parser.add_argument('--saved_model_dir',
                        type=str,
                        dest="saved_mode_dir",
                        help="Where to save the exported graph")
    parser.add_argument('--final_tensor_name',
                        type=str,
                        default='final_tensor_name',
                        help="""The name of the output classification layer in the retrained graph.""")
    parser.add_argument('--output_graph',
                        type=str,
                        default='/tmp/retrained_graph.pb',
                        help="""The name of the output classification layer in the retrained graph.""")
    parser.add_argument('--intermediate_store_frequency',
                        default=100,
                        type=int,
                        dest="intermediate_store_frequency",
                        help="When to store the intermediate graphs")
    parser.add_argument('--intermediate_output_graph_dir',
                        type=str,
                        default='/tmp/intermediate_graphs/',
                        dest="intermediate_output_graph_dir",
                        help="Where to stor the intermediate graphs")
    parser.add_argument('--eval_step_interval',
                        type=int,
                        default=10,
                        dest='eval_step_interval',
                        help="""How often to evaluate the training results""")
    flags, unparsed_args = parser.parse_known_args()
    return flags, unparsed_args


class ImageDataset:
    def __init__(self, flags):
        self.image_dir = flags.image_dir
        self.test_percent = flags.testing_percentage
        self.valid_percent = flags.validation_percentage
        self.gt_file = flags.labels_file
        self.file_list = {}
        self.results = {}
        self.create_img_list()

    def create_img_list(self):
        if not gfile.Exists(self.image_dir):
            tf.logging.error("Image directory '" + self.image_dir + "' not found")
            return

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list: List[Path] = []
        labels = {}

        # TODO: convert to list comprehension
        for extension in extensions:
            file_list.extend(list(Path(self.image_dir).rglob(f"*.{extension}")))

        self.file_list = {file_path.name: str(file_path) for file_path in file_list}

        if not file_list:
            raise RuntimeError('No files found')

        with open(self.gt_file, 'r') as csvfile:
            label_file = csv.reader(csvfile, delimiter='\t')
            for row in label_file:
                labels[row[0]] = [float(value) for value in row[1:]] 
                # labels.append([float(values) for values in row[1:]])  # TODO: convert to list comprehension

        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', str(file_name))

            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (VAL)) * (100.0 / VAL))
            if percentage_hash < self.valid_percent:
                validation_images.append(base_name)
            elif percentage_hash < (self.test_percent + self.valid_percent):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        print('No. of training images : ', len(training_images))
        print('No. of validation images : ', len(validation_images))
        print('No. of testing images : ', len(testing_images))

        self.results = {  # TODO: find a better name
            'validation': validation_images,
            'training': training_images,
            'testing': testing_images,
            'labels': labels
        }

    def get_path(self, index, category):
        if category not in self.results:
            tf.logging.fatal('Category does not exists %s', category)
        category_list = self.results[category]
        if not category_list:
            tf.logging.fatal('Category %s has no images', category)
        base_name = category_list[index]
        return self.file_list[base_name]
        # full_path = os.path.join(self.image_dir, base_name)
        # return full_path

    def get_ground_truth(self, index, category):
        img_name = self.get_path(index, category)
        label_index = os.path.basename(img_name).split('.')[0]
        ground_truth = self.results['labels'][label_index]
        return ground_truth

    def get_list(self):
        return self.results


class Bottlenecks:
    def __init__(
            self,
            sess,
            image_data,
            bdir,
            architecture,
            jpeg_data_tensor,
            decoded_image_tensor,
            resized_input_tensor,
            bottleneck_tensor
        ):
        self.sess = sess
        self.image_dataset = image_data
        self.bdir = bdir
        self.architecture = architecture
        self.jpeg_data_tensor = jpeg_data_tensor
        self.decoded_image_tensor = decoded_image_tensor
        self.resized_input_tensor = resized_input_tensor
        self.bottleneck_tensor = bottleneck_tensor

        self.cache()

    def run_on_image(self, image_data):
        # value = self.sess.run(self.bottleneck_tensor, feed_dict={self.resized_input_tensor: image_data})
        value = self.sess.run(self.decoded_image_tensor, feed_dict={self.jpeg_data_tensor: image_data})
        output_image = self.sess.run(self.bottleneck_tensor, feed_dict={self.resized_input_tensor: value})
        return np.squeeze(output_image)

    def create_file(self, path, index, category):
        tf.logging.info('creating bottleneck at ' + path)
        image_path = self.image_dataset.get_path(index, category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('file does not exists %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        try:
            value = self.run_on_image(image_data)
        except Exception as exep:
            raise RuntimeError('error during processing file %s (%s)' % (image_path, str(exep)))

        # bstring = np.array_str(value)
        # bstring = ','.join(str(x) for x in value)
        np.save(path, value, allow_pickle=False)
        # with open(path, 'w') as bfile:
        #     bfile.write(bstring)

    def get_path(self, index, category):
        if category not in self.image_dataset.results:
            tf.logging.fatal('category does not exists %s', category)
        category_list = self.image_dataset.results[category]
        if not category_list:
            tf.logging.fatal('category %s has no images', category)

        base_name = category_list[index]
        full_path = os.path.join(self.bdir, base_name) + '_' + self.architecture + '.txt'
        return full_path

    def get_or_create(self, index, category):
        path = self.get_path(index, category)

        if not os.path.exists(f"{path}.npy"):
            self.create_file(path, index, category)

        bvalues = np.load(f"{path}.npy")
        # with open(path, 'r') as bfile:
        #     bstring = bfile.read()

        # did_hit_error = False
        # try:
        #     bvalues = np.fromstring(bstring)
        # except ValueError:
        #     tf.logging.warning('inavlid float found, recreating bottlenecks')
        #     did_hit_error = True

        # if did_hit_error:
        #     self.create_file(path, index, category)

        # with open(path, 'r') as bfile:
        #     bstring = bfile.read()

        # bvalues = [float(x) for x in bstring.split(',')]
        return bvalues

    def cache(self):
        how_many = 0
        for category in ['training', 'testing', 'validation']:
            category_list = self.image_dataset.results[category]
            for index, unused_base_name in enumerate(category_list):
                self.get_or_create(index, category)
                how_many += 1
                if how_many % 1000 == 0:
                    tf.logging.info(str(how_many) + 'bottlenecks files created')

    def get_rand_cached(self, how_many, category):
        bottlenecks = []
        ground_truth = []

        if how_many >= 0:
            category_list = self.image_dataset.results[category]
            for unused_i in range(how_many):
                image_index = random.randrange(0, len(category_list), 1)
                bottlenecks.append(self.get_or_create(image_index, category))
                ground_truth.append(self.image_dataset.get_ground_truth(image_index, category))
        else:
            for image_index in len(self.image_dataset.results[category]):
                bottlenecks.append(self.get_or_create(image_index, category))
                ground_truth.append(self.image_dataset.get_ground_truth(image_index, category))
        return bottlenecks, np.array(ground_truth)


class Model:
    def __init__(self, flags):
        self.flags = flags
        self.model_info = {}
        self.eval_graph = None
        self._bottlenecks = None
        self.resized_input_tensor = None

        if tf.gfile.Exists(self.flags.summaries_dir):
            tf.gfile.DeleteRecursively(self.flags.summaries_dir)
            tf.gfile.MakeDirs(self.flags.summaries_dir)
        # if self.flags.intermediate_store_frequency > 0:
        #     self.create_dest_dir(self.flags.intermediate_output_graph_dir)

        self.create_model_info()

        if not self.model_info:
            # tf.logging.error('Did not recognize the architecture flag')
            raise RuntimeError('Did not recognize the architecture flag')

            # return -1 ## Add a value error over here

        self.filename = self.model_info['data_url'].split('/')[-1]
        self.filepath = os.path.join(self.flags.model_dir, self.filename)

        self.download_extract_if_needed()
        [self.graph, self.bottleneck_tensor, self.resized_input_tensor] = self.create_model_graph()

    @staticmethod
    def create_dest_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def download(self):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (self.filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(self.model_info['data_url'], self.filepath, _progress)  #### filepath clobbered !!!
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded %s %d bytes.', self.filename, statinfo.st_size)

    def extract(self):
        print('Extracting file from ', self.filepath)
        tarfile.open(self.filepath, 'r:gz').extractall(self.flags.model_dir)

    def download_extract_if_needed(self):
        if os.path.exists(self.filepath):
            print('Not extracting or downloading files, model already present in disk')
        else:
            dirname = self.filepath
            self.create_dest_dir(self.filepath)
            self.download()
            self.extract()

    def create_model_info(self):
        architecture = self.flags.architecture.lower()
        is_quantized = False
        if architecture == 'inception_v3':
            model_file_name = 'classify_image_graph_def.pb'
            data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
            resized_input_tensor_name = 'Mul:0'
            bottleneck_tensor_name = 'pool_3/_reshape:0'
            bottleneck_tensor_size = 2048
            input_width = 299
            input_height = 299
            input_depth = 3
            input_mean = 128
            input_std = 128


        elif architecture.startswith('mobilenet_'):
            parts = architecture.split('_')
            if len(parts) != 3 and len(parts) != 4:
                tf.logging.error("Couldn't understand architecture name '%s'", architecture)
                return

            version_string = parts[1]
            if (version_string not in ['1.0', '0.75', '0.5', '0.25']):
                tf.logging.error(
                    """The Mobilenet version should be '1.0', '0.75', '0.5', or '0.25', but found '%s' for architecture '%s'""",
                    version_string, architecture
                )
                return

            size_string = parts[2]
            if (size_string not in ['224', '192', '160', '128']):
                tf.logging.error(
                    """The Mobilenet input size should be '224', '192', '160', or '128', but found '%s' for architecture '%s'""",
                    size_string, architecture
                )
                return

            is_quantized = (len(parts) != 3)
            if not is_quantized:
                if parts[3] != 'quant':
                    tf.logging.error("Couldn't understand architecture suffix '%s' for '%s'", parts[3], architecture)
                    return

            data_url = 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/'
            model_name = 'mobilenet_v1_' + version_string + '_' + size_string

            if is_quantized:
                model_name += '_quant'

                model_file_name = model_name + '_frozen.pb'
                data_url += model_name + '.tgz'
                resized_input_tensor_name = 'input:0'
                bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
                bottleneck_tensor_size = 1001
                input_width = int(size_string)
                input_height = int(size_string)
                input_depth = 3
                input_mean = 127.5
                input_std = 127.5
        else:
            tf.logging.error("Couldn't understand architecture name '%s'", architecture)
            raise ValueError('Unknown architecture', architecture)

        self.model_info = {
            'data_url': data_url,
            'bottleneck_tensor_name': bottleneck_tensor_name,
            'bottleneck_tensor_size': bottleneck_tensor_size,
            'input_width': input_width,
            'input_height': input_height,
            'resized_input_tensor_name': resized_input_tensor_name,
            'input_depth': input_depth,
            'model_file_name': model_file_name,
            'input_mean': input_mean,
            'input_std': input_std,
            'quantize_layer': is_quantized,
        }

    def create_model_graph(self):
        if not self.model_info:
            tf.logging.error('Did not recognize architecture flag')
            return -1

        with tf.Graph().as_default() as graph:
            model_path = os.path.join(FLAGS.model_dir, self.model_info['model_file_name'])
            print('Model path: ', model_path)
            with gfile.FastGFile(model_path, 'rb') as f_handle:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f_handle.read())
                bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
                    graph_def,
                    name='',
                    return_elements=[
                        self.model_info['bottleneck_tensor_name'],
                        self.model_info['resized_input_tensor_name']
                    ]))

        return graph, bottleneck_tensor, resized_input_tensor

    @staticmethod
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def add_jpeg_decoding(self):
        jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
        decoded_image = tf.image.decode_jpeg(jpeg_data, channels=self.model_info['input_depth'])
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([self.model_info['input_height'], self.model_info['input_width']])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
        offset_image = tf.subtract(resized_image, self.model_info['input_mean'])
        mul_image = tf.multiply(offset_image, 1.0 / self.model_info['input_std'])
        return jpeg_data, mul_image

    def add_final_retrain_ops(self, is_training=True):
        bottleneck_tensor_size = self.model_info['bottleneck_tensor_size']
        with tf.name_scope('input'):
            bottleneck_input = tf.placeholder_with_default(self.bottleneck_tensor, shape=[None, bottleneck_tensor_size],
                                                           name="BottleneckInputPlaceholder")

            ground_truth_input = [
                tf.placeholder(tf.float32, shape=[None, 1], name='roll_GroundTruthInput'),
                tf.placeholder(tf.float32, shape=[None, 1], name='yaw_GroundTruthInput'),
                tf.placeholder(tf.float32, shape=[None, 1], name='pitch_GroundTruthInput')
            ]

        with tf.name_scope('final_retrain_ops'):
            with tf.name_scope('weights'):
                roll_layer_weights = tf.Variable(tf.truncated_normal([bottleneck_tensor_size, 1], stddev=0.001),name='roll_final_weights')
                pitch_layer_weights = tf.Variable(tf.truncated_normal([bottleneck_tensor_size, 1], stddev=0.001),name='pitch_final_weights')
                yaw_layer_weights = tf.Variable(tf.truncated_normal([bottleneck_tensor_size, 1], stddev=0.001),name='yaw_final_weights')

                # tf.summary.scalar(roll_layer_weights)
                # tf.summary.scalar(pitch_layer_weights)
                # tf.summary.scalar(yaw_layer_weights)

            with tf.name_scope('biases'):
                roll_layer_biases = tf.Variable(tf.zeros([1]), name='final_biases')
                pitch_layer_biases = tf.Variable(tf.zeros([1]), name='final_biases')
                yaw_layer_biases = tf.Variable(tf.zeros([1]), name='final_biases')

            with tf.name_scope('Wx_plus_b'):
                roll_logits = tf.matmul(bottleneck_input, roll_layer_weights) + roll_layer_biases
                pitch_logits = tf.matmul(bottleneck_input, pitch_layer_weights) + pitch_layer_biases
                yaw_logits = tf.matmul(bottleneck_input, yaw_layer_weights) + yaw_layer_biases

                tf.summary.histogram('yaw_regression', yaw_logits)
                tf.summary.histogram('pitch_regression', pitch_logits)
                tf.summary.histogram('roll_regressions', roll_logits)

                last_layer = tf.concat([roll_logits, yaw_logits, pitch_logits], axis=1, name=FLAGS.final_tensor_name)

            if self.model_info['quantize_layer']:
                if is_training:
                    tf.contrib.quantize.create_training_graph()
                else:
                    tf.contrib.quantize.create_eval_graph()

            if not is_training:
                return None, None, bottleneck_input, ground_truth_input, last_layer

            with tf.name_scope('Yaw_MSE'):
                yaw_mse_loss = tf.losses.mean_squared_error(labels=ground_truth_input[1], predictions=yaw_logits)
                tf.summary.scalar('Yaw_Error', yaw_mse_loss)

            with tf.name_scope('Pitch_MSE'):
                pitch_mse_loss = tf.losses.mean_squared_error(labels=ground_truth_input[2], predictions=pitch_logits)
                tf.summary.scalar('Pitch_Error', pitch_mse_loss)

            with tf.name_scope('Roll_MSE'):
                roll_mse_loss = tf.losses.mean_squared_error(labels=ground_truth_input[0], predictions=roll_logits)
                tf.summary.scalar('Roll_Error', roll_mse_loss)

            with tf.name_scope('train'):
                roll_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                roll_train_step = roll_optimizer.minimize(roll_mse_loss,
                                                          var_list=[roll_layer_weights, roll_layer_biases])

                yaw_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                yaw_train_step = yaw_optimizer.minimize(yaw_mse_loss, var_list=(yaw_layer_weights, yaw_layer_biases))

                pitch_optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
                pitch_train_step = pitch_optimizer.minimize(pitch_mse_loss,
                                                            var_list=(pitch_layer_weights, pitch_layer_biases))

            train_step = [roll_train_step, yaw_train_step, pitch_train_step]
            loss = [roll_mse_loss, yaw_mse_loss, pitch_mse_loss]

            return train_step, loss, bottleneck_input, ground_truth_input, last_layer

    @staticmethod
    def add_eval_step(result_tensor, ground_truth_tensor):
        with tf.name_scope('correlations'):
            with tf.name_scope('Yaw_correlation'):
                yaw_correlation, yaw_update_op = streaming_pearson_correlation(predictions=result_tensor[:, 0],
                                                                               labels=ground_truth_tensor[0])
                tf.summary.scalar('Yaw_correlation', yaw_update_op)

            with tf.name_scope('Roll_correlation'):
                roll_correlation, roll_update_op = streaming_pearson_correlation(predictions=result_tensor[:, 1],
                                                                                 labels=ground_truth_tensor[1])
                tf.summary.scalar('Roll_correlation', roll_update_op)

            with tf.name_scope('Pitch_correlation'):
                pitch_correlation, pitch_update_op = streaming_pearson_correlation(predictions=result_tensor[:, 2],
                                                                                   labels=ground_truth_tensor[2])
                tf.summary.scalar('Pitch_correlation', pitch_update_op)

            final_correlations = [yaw_correlation, roll_correlation, pitch_correlation]
            streaming_correlations = [yaw_update_op, roll_update_op, pitch_update_op]

        return streaming_correlations, final_correlations

    def build_eval_sess(self):
        if not self.eval_graph:
            self.eval_graph, self.bottleneck_tensor, _ = self.create_model_graph()

        eval_sess = tf.Session(graph=self.eval_graph)
        with self.eval_graph.as_default():
            (_, _, bottleneck_input, ground_truth_input, last_layer) = self.add_final_retrain_ops(False)

            tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)
            streaming_correlations, final_correlations = self.add_eval_step(last_layer, ground_truth_input)

        return eval_sess, bottleneck_input, ground_truth_input, last_layer, streaming_correlations, final_correlations

    def run_final_eval(self):
        sess, bottleneck_input, ground_truth_input, last_layer, _, _ = self.build_eval_sess()

        test_bottlenecks, test_ground_truths = self.bottlenecks.get_rand_cached(self.flags.test_batch_size, 'testing')

        test_correlation_values = sess.run(
            last_layer,
            feed_dict={
                bottleneck_input: test_bottlenecks,
                ground_truth_input[0]: test_ground_truths[:, 0].reshape(-1, 1),
                ground_truth_input[1]: test_ground_truths[:, 1].reshape(-1, 1),
                ground_truth_input[2]: test_ground_truths[:, 2].reshape(-1, 1)
            }
        )

        average_test_correlation = tf.reduce_mean(test_correlation_values)
        tf.logging.info('Final average test correlation = %.1f' % (average_test_correlation))

    def save_graph(self, file_name):
        sess, _, _, _, _, _ = self.build_eval_sess()
        graph = sess.graph

        output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [
            'final_retrain_ops/Wx_plus_b/' + self.flags.final_tensor_name])

        with gfile.FastGFile(file_name, 'wb') as f_handle:
            f_handle.write(output_graph_def.SerializeToString())

    @property
    def bottlenecks(self):
        return self._bottlenecks

    @bottlenecks.setter
    def set_bottlenecks(self, bottlenecks):
        self._bottlenecks = bottlenecks


def main_setup(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    model = Model(FLAGS)
    images = ImageDataset(FLAGS)

    with model.graph.as_default():
        train_step, mse_loss, bottleneck_input, ground_truth_input, last_layer = model.add_final_retrain_ops(
            is_training=True)

    with tf.Session(graph=model.graph) as sess:
        jpeg_data_tensor, decoded_image_tensor = model.add_jpeg_decoding()

        bottlenecks = Bottlenecks(
            sess=sess,
            image_data=images,
            bdir=model.flags.bottleneck_dir,
            architecture=model.flags.architecture,
            jpeg_data_tensor=jpeg_data_tensor,
            decoded_image_tensor=decoded_image_tensor,
            resized_input_tensor=model.resized_input_tensor,
            bottleneck_tensor=model.bottleneck_tensor
        )

        model.set_bottlenecks = bottlenecks

        streaming_correlations, final_correlations = model.add_eval_step(last_layer, ground_truth_input)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation', sess.graph)

        train_saver = tf.train.Saver()

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        for i in range(FLAGS.how_many_training_steps):
            print("reached_step: %d" % (i))
            train_bottlenecks, train_ground_truth = model.bottlenecks.get_rand_cached(FLAGS.train_batch_size,
                                                                                      'training')
            train_summary, _ = sess.run(
                [merged, mse_loss],
                feed_dict={
                    bottleneck_input: np.array(train_bottlenecks),
                    ground_truth_input[0]: train_ground_truth[:, 0].reshape(-1, 1),
                    ground_truth_input[1]: train_ground_truth[:, 1].reshape(-1, 1),
                    ground_truth_input[2]: train_ground_truth[:, 2].reshape(-1, 1)
                }
            )
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval == 0) or is_last_step:
                (yaw_correlation, roll_correlation, pitch_correlation), (yaw_loss, roll_loss, pitch_loss) = sess.run(
                    [streaming_correlations, mse_loss],
                    feed_dict={
                        bottleneck_input: train_bottlenecks,
                        ground_truth_input[0]: train_ground_truth[:, 0].reshape(-1, 1),
                        ground_truth_input[1]: train_ground_truth[:, 1].reshape(-1, 1),
                        ground_truth_input[2]: train_ground_truth[:, 2].reshape(-1, 1)
                    }
                )
                tf.logging.info('%s: Step %d: average loss:%.1f average correlation = %.1f' % (
                datetime.now(), i, (yaw_loss + roll_loss + pitch_loss) / 3,
                (yaw_correlation + roll_correlation + pitch_correlation) / 3))

                validation_bottlenecks, validation_ground_truth = model.bottlenecks.get_rand_cached(
                    model.flags.validation_batch_size, 'validation')
                validation_summary, correlation_values = sess.run(
                    [merged, streaming_correlations],
                    feed_dict={
                        bottleneck_input: validation_bottlenecks,
                        ground_truth_input[0]: validation_ground_truth[:, 0].reshape(-1, 1),
                        ground_truth_input[1]: validation_ground_truth[:, 1].reshape(-1, 1),
                        ground_truth_input[2]: validation_ground_truth[:, 2].reshape(-1, 1)
                    }
                )

                average_validation_correlation = np.mean(correlation_values)

                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation correlation = %1.1f (N=%d)' % (
                datetime.now(), i, average_validation_correlation, len(validation_bottlenecks)))

            intermediate_frequency = FLAGS.intermediate_store_frequency

            if (intermediate_frequency > 10 and (i % intermediate_frequency == 0) and i > 0):
                train_saver.save(sess, CHECKPOINT_NAME)
                intermediate_file_name = (FLAGS.intermediate_output_graph_dir + 'intermediate' + str(i) + '.pb')
                model.save_graph(intermediate_file_name)
        train_saver.save(sess, CHECKPOINT_NAME)

        test_bottlenecks, test_ground_truths = model.bottlenecks.get_rand_cached(FLAGS.test_batch_size, 'testing')

        final_answers, rotation_vals = sess.run(
            [final_correlations, last_layer],
            feed_dict={
                bottleneck_input: test_bottlenecks,
                ground_truth_input[0]: test_ground_truths[:, 0].reshape(-1, 1),
                ground_truth_input[1]: test_ground_truths[:, 1].reshape(-1, 1),
                ground_truth_input[2]: test_ground_truths[:, 2].reshape(-1, 1)
            }
        )

        print('final_answers: ', np.sum(final_answers) / 3)
        print('rotation_vals: ', rotation_vals)
        tf.logging.info("Saving final graph at: " + FLAGS.output_graph)
        model.save_graph(FLAGS.output_graph)


if __name__ == "__main__":
    FLAGS, UNPARSED = parse_arguments()
    tf.app.run(main=main_setup, argv=[sys.argv[0]] + UNPARSED)
