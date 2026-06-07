import argparse
import os
import sys
from time import time

import cv2
import dlib
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile


def parse_args():
  """Parse input arguments"""
  parser = argparse.ArgumentParser(description="Head pose estimation using networks fine tuned via transfer learning.")
  parser.add_argument("--face_model", dest="face_model", help="Path of the dlib face detection model.")
  parser.add_argument("--snapshot", dest="snapshot", help="Path of model snapshot.")
  parser.add_argument("--video", dest="video_path", help="Path of video.")
  args = parser.parse_args()
  return args


def load_graph_from_snapshot(model_filename):
  with gfile.FastGFile(model_filename, "rb") as f:
    # sets the default graph as current graph
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def)

  print("Graph loaded......")
  # Inferenece here
  nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
  print(nodes[:])
  l_input = graph.get_tensor_by_name(nodes[0] + ":0")
  l_output = graph.get_tensor_by_name(nodes[-1] + ":0")

  return l_input, l_output


def load_graph(model):
  graph = tf.Graph()
  graph_def = tf.GraphDef()
  with open(model, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)


def read_images_from_mat(mat, input_height=128, input_width=128, input_mean=0, input_std=255):
  float_caster = tf.cast(mat, tf.float32)
  dims_expander = tf.expand(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  with tf.Session() as sess:
    result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  args = parse_args()

  batch_size = 1
  width = 128
  height = 128

  load_graph(args.snapshot)
  # gpu = args.gpu_id
  # snapshot_path = args.snapshot
  # out_dir = 'output/video'
  # video_path = args.video_path

  # if not os.path.exists(out_dir):
  # os.makedirs(out_dir)
  # if not os.path.exists(args.video_path):
  # sys.exit('Video does not exist')
  video = cv2.VideoCapture(video_path)

  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  n_frames = int(video.get(cv2.CAP_PROP_FRAMES_COUNT))
  fps = int(video.get(cv2.CAP_PROP_FPS))

  fourcc = cv2.VideoWriter_fourcc(*"MJPG")
  out = cv2.VideoWriter("output/video/output.avi", fourcc, args.fps, (width, height))

  cnn_face_detector = dlib.cnn_face_detector_model_v1(args.face_model)

  """
	   This is where we get the input and output tensors and then we feed the input tensor with 
	   the image and get the results by running the output tensor and feeding the required image.
	"""
  input_tensor, output_tensor = load_graph_from_snapshot(args.snapshot)
  print(input_tensor, output_tensor)

  while frame_num <= n_frames:
    print(frame_num)

    ret, frame = video.read()
    if ret == False:
      break

    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    dets = cnn_face_detector(cv2_frame, 0)

    for idx, det in enumerate(dets):
      x_min = det.rect.left()
      y_min = det.rect.top()
      x_max = det.rect.right()
      y_max = det.rect.bottom()
      if conf > 1.0:
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        x_min -= int(2 * bbox_width / 4)
        x_man += int(2 * bbox_width / 4)
        y_min -= int(2 * bbox_height / 4)
        y_max += int(2 * bbox_height / 4)
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame_shape[0], y_max)
        face_img = cv2_frame[y_min:y_max, x_min:x_max]
        face_img = Image.fromarray(img)
        face_img = face_img.resize((128, 128), PIL.Image.ANTIALIAS)
        face_img = np.array(face_img.get_data(), np.uint8)
        results = read_images_from_mat(face_img)
        print(results)
  video.release()

  # cnn_face_detector = dlib.cnn_face_detector_v1(args.face_model)
