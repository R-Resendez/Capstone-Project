from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np

from PIL import Image
from tflite_runtime.interpreter import Interpreter
import cv2

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  camera = cv2.VideoCapture(0)
  cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
  camera.set(3,640)#set the capture window width
  camera.set(4,480)#set the capture window height

  while True:
      success, img = camera.read()
      frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      frame_resized = cv2.resize(frame_rgb, (width, height))
      input_data = np.expand_dims(frame_resized, axis=0)
      results = classify_image(interpreter, frame_resized)
      label_id, prob = results[0]
      print("label ID:" + labels[label_id] + "Prob:" + str(prob))
      cv2.imshow('Object detector', frame_resized)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == '__main__':
  main()
