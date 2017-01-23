'''
ConvNet based handwritten digits recognition on MNIST data set.
Accuracy ~99%
'''
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data

# conv 1
filter_size1 = 5      # 5x5 filter dimension
num_filters1 = 16     # 16 filters in layer 1

# conv 2
filter_size2 = 5
num_filters2 = 36

# fully connected layer
fc_size = 128

data = input_data.read_data_sets("data/MNIST/", one_hot=True)

# convert test classes values from one-hot coding to 0-9 integers
data.test.cls = np.argmax(data.test.labels, axis=1)

# actual image dimension is 28x28
img_size = 28
# convert it into an input vector of 28*28 = 784 dimension
img_size_flat = img_size * img_size

img_shape = (img_size, img_size)

# 1 channel for grayscale
num_channels = 1

# number of classes 0-9
num_classes = 10


# tensorflow computation graph begins

# helper functions for creating new weights and biases
def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):

	# shape of the each filter-weights
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	# create new weights
	weights = new_weights(shape=shape)

	# create new biases
	biases = new_biases(length=num_filters)

	# convolution operation
	# strides = 1, padding = 1 (to maintain spatial size same as previous layer)
	layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')

	# add biases to the results of convolution to each filter
	layer += biases

	if use_pooling:
		# 2x2 max-pooling
		layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# ReLU operation, max(x, 0)
	layer = tf.nn.relu(layer)

	return layer, weights

# flatten layer for fully connected neural net
def flatten_layer(layer):
	# get shape of the input layer
	layer_shape = layer.get_shape()

	# layer shape is of the form [num_images, img_height, img_width, num_channels]
	# num_features = img_height*img_width*num_channels
	num_features = layer_shape[1:4].num_elements()

	layer_flat = tf.reshape(layer, [-1, num_features])

	return layer_flat, num_features

# create fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
	# weights and biases for fc layer
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_biases(length=num_outputs)

	# linear operation
	layer = tf.matmul(input, weights) + biases

	if use_relu:
		layer = tf.nn.relu(layer)
	
	return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
# conv layers require image to be in shape [num_images, img_height, img_weight, num_channels]
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# conv layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
											num_input_channels=num_channels,
											filter_size=filter_size1,
											num_filters=num_filters1,
											use_pooling=True)

# conv layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
											num_input_channels=num_filters1,
											filter_size=filter_size2,
											num_filters=num_filters2,
											use_pooling=True)

# flatten layer
layer_flat, num_features = flatten_layer(layer_conv2)

# fully connected layer 1
layer_fc1 = new_fc_layer(input=layer_flat,
						num_inputs=num_features,
						num_outputs=fc_size,
						use_relu=True)

# fully connected layer 2
layer_fc2 = new_fc_layer(input=layer_fc1,
						num_inputs=fc_size,
						num_outputs=num_classes,
						use_relu=False)

# predicted class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# cost function 
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64

total_iterations = 0

def optimize(num_iterations):
	global total_iterations

	start_time = time.time()

	for i in range(total_iterations, total_iterations + num_iterations):

		x_batch, y_true_batch = data.train.next_batch(train_batch_size)
		feed_dict_train = {x: x_batch, y_true: y_true_batch}

		session.run(optimizer, feed_dict=feed_dict_train)

		# print status after every 100 iterations
		if i % 100 == 0:
			# calculate accuracy on the training set
			acc = session.run(accuracy, feed_dict= feed_dict_train)

			msg = "Optimization iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
			print msg.format(i+1, acc)

	# update total iterations performed
	total_iterations += num_iterations

	# ending time
	end_time = time.time()
	time_dif = end_time - start_time

	print "time usage: "+ str(timedelta(seconds=int(round(time_dif))))

optimize(1000)