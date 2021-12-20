import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

#################################
# Sets the label color to red or black 
# for determining if the model did guess correctly or not.
#################################

def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'


# Set the local pathway to the training files that were used for training the model.
# Then the data is loaded into the model_maker data loader.
image_path = ""
image_path = os.path.join(os.path.dirname(image_path), 'Training_Images')
data = DataLoader.from_folder(image_path)


#Simply take that data and split it up between a training, testing and validation datasets.
train_data, test_data = data.split(0.9)
validation_data, test_data = test_data.split(0.5)

# Take the data that was just loaded and load a sample onto a doc for visual validation.
# This code was taken directly from google's model_maker example as I felt it was a great way to easily verify my results.
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)
  plt.xlabel(data.index_to_label[label.numpy()])
plt.show()


# This is where the model is created from the mobilenet cnn from google.
# model = image_classifier.create(train_data, validation_data=validation_data)
model = image_classifier.create(train_data, model_spec=model_spec.get('mobilenet_v2'), validation_data=validation_data)


# Now we can take a look at the model summary and then we can go ahead and evaluate it to get an udnerstanding of total loss and accuracy.
model.summary()
loss, accuracy = model.evaluate(test_data)


# This is another section of code that was pulled directly from the example on transfer training for TF lite model maker.
# Here we make predicitons based on the images provided in the test dataset.
# From there the images are then iterated through and put onto a page that will allow me to visually confirm the accuracy of each image in the test dataset.
plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image.numpy(), cmap=plt.cm.gray)

  predict_label = predicts[i][0][0]
  color = get_label_color(predict_label,
                          test_data.index_to_label[label.numpy()])
  ax.xaxis.label.set_color(color)
  plt.xlabel('Predicted: %s' % predict_label)
plt.show()


# Now that the model has been trained and validated we can go ahead and export both the model as well as the label file.
model.export(export_dir='.')
model.export(export_dir='.', export_format=ExportFormat.LABEL)

model.evaluate_tflite('model.tflite', test_data)