'''
A basic linear model for MNIST digits classification problem
Maximum accuracy so far is 92.7%
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("data/MNIST/", one_hot=True)

data.test.cls = np.array([label.argmax() for label in data.test.labels])

img_size = 28
img_size_flat = img_size * img_size # 784

img_shape = (img_size, img_size)

num_classes = 10 # 0 to 9

# helper function for plotting images

def plot_images(images, cls_true, cls_pred=None):
	assert len(images) == len(cls_true) == 9

	# create figure with 3x3 sub-plots
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# plot image
		ax.imshow(images[i].reshape(img_shape), cmap='binary')

		# show true and predicted classes
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		ax.set_xlabel(xlabel)

		# remove ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

# get the first images from the test-set
images = data.test.images[0:9]

# get the true classes for those images
cls_true = data.test.cls[0:9]

# plot the images and labels
#plot_images(images=images, cls_true=cls_true)


# create the computation graph
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

logits = tf.matmul(x, weights) + biases

y_pred = tf.nn.softmax(logits)

y_pred_cls = tf.argmax(y_pred, dimension=1)

# cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# using gradient descent for parameters optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100

feed_dict_test = {x: data.test.images, y_true: data.test.labels, y_true_cls: data.test.cls}

def optimize(num_iterations):
	for i in range(num_iterations):
		x_batch, y_true_batch = data.train.next_batch(batch_size)

		feed_dict_train = {x: x_batch, y_true: y_true_batch}

		session.run(optimizer, feed_dict=feed_dict_train)
		print_accuracy()

def print_accuracy():

	acc = session.run(accuracy, feed_dict=feed_dict_test)
	print "accuracy on test-set: {0:.1%}".format(acc)

optimize(num_iterations=10000)
