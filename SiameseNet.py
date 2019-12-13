'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for more details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
# import tensorflow.compat.v1 as tf
import tensorflow as tf

import keras.losses

# from tensorflow import keras


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
  """
  Freezes the state of a session into a pruned computation graph.

  Creates a new computation graph where variable nodes are replaced by
  constants taking their current value in the session. The new graph will be
  pruned so subgraphs that are not necessary to compute the requested
  outputs are removed.
  @param session The TensorFlow session to be frozen.
  @param keep_var_names A list of variable names that should not be frozen,
                        or None to freeze all the variables in the graph.
  @param output_names Names of the relevant graph outputs.
  @param clear_devices Remove the device directives from the graph for better portability.
  @return The frozen graph definition.
  """
  graph = session.graph
  with graph.as_default():
    freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
    output_names = output_names or []
    output_names += [v.op.name for v in tf.global_variables()]
    input_graph_def = graph.as_graph_def()
    if clear_devices:
      for node in input_graph_def.node:
        node.device = ""
    frozen_graph = tf.graph_util.convert_variables_to_constants(
      session, input_graph_def, output_names, freeze_var_names)
    return frozen_graph


num_classes = 10
epochs = 1

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
pb_name = tf.train.write_graph(frozen_graph, "exports", "siamese1.pb", as_text=False)

# load graph again
with tf.gfile.GFile(pb_name, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

vars = ['dense_1/kernel', 'dense_1/bias', 'dense_2/kernel', 'dense_2/bias', 'dense_3/kernel', 'dense_3/bias', 'lambda_1/Sqrt']

# mkae subgraph --> remove nodes
subgraph = tf.graph_util.extract_sub_graph(graph_def, vars)
tf.train.write_graph(subgraph, "exports", "siamese_2.pb", as_text=False)
print('poopy')

# ['lambda_1/Sqrt', 'RMSprop/learning_rate', 'RMSprop/rho', 'RMSprop/decay', 'RMSprop/iterations', 'total', 'count', 'training/RMSprop/accumulator_0_1', 'training/RMSprop/accumulator_1_1', 'training/RMSprop/accumulator_2_1', 'training/RMSprop/accumulator_3_1', 'training/RMSprop/accumulator_4_1', 'training/RMSprop/accumulator_5_1']


# name = "siamese_1"
# model.save(name + "h5", save_format="tf")
# # tf.keras.models.save_model(name + ".h5", save_format="tf")
# model_saved = keras.models.load_model(name + ".h5", custom_objects={'contrastive_loss': contrastive_loss})
# keras.experimental.export_saved_model(model_saved, 'exports/' + name + ".pb")


