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
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import tensorflow as tf
from get_dataset import get_fashion_mnist
import keras.losses
from keras.models import load_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

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
    '''
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(Params.num_classes)]) - 1
    for d in range(Params.num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, Params.num_classes)
            dn = (d + inc) % Params.num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    # base_network = keras.applications.mobilenet.MobileNet(include_top=False,
    #                                                       weights='imagenet',
    #                                                       # models/mobilenet_siamese_weights_3epoch.h5
    #                                                       alpha=0.25,
    #                                                       input_tensor=input)
    base_network = keras.applications.densenet.DenseNet121(
                                                  include_top=False,
                                                  weights='imagenet',
                                                  input_tensor=input
                                                  )

    base_network.trainable = True
    base_network.layers[-1].outbound_nodes = []
    base_network.outputs = [base_network.layers[-1].output]
    output = base_network.outputs
    output = Flatten()(output)
    output = Dense(output_dim=Params.dense_nodes, activation='relu')(output)

    return Model(base_network.input, output)


def create_base_network2(input_shape):
  '''Base network to be shared (eq. to feature extraction).
  '''
  input = Input(shape=input_shape)
  x = Conv2D(filters=32, kernel_size=7, strides=(1,1), activation='relu', input_shape = input_shape, kernel_regularizer=regularizers.l2(0.01))(input)
  x = MaxPool2D((2,2))(x)
  x = Conv2D(filters=64, kernel_size=4, strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = MaxPool2D((2,2))(x)
  x = Conv2D(filters=64, kernel_size=3, strides=(1,1), activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
  x = MaxPool2D((2,2))(x)
  x = Conv2D(filters=64, kernel_size=3, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
  x = Flatten()(x)
  x = Dense(256)(x)

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


class Params:
  imsize = 64
  values = None
  num_classes = 10
  epochs = 6
  dense_nodes = 128
  load_model = False
  model_name = "mobilenet_siamese_3epoch"

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = get_fashion_mnist(imsize=Params.imsize)

input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(Params.num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(Params.num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

# network definition
x_train = None
x_test = None

# load model if required
if Params.load_model:
  base_network = load_model(f'models/{Params.model_name}.h5')
else:
  # base_network = create_base_network(input_shape=input_shape)
  base_network = create_base_network2(input_shape=input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

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
          epochs=Params.epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

# save the model
# model.save("models/mobilenet_siamese_6epoch.h5")
# model.save_weights('superTrainedSiamese.h5')
# yaml_string = model.to_yaml()
# with open(r'models/mobilenet_siamese_architecture_6epoch.yaml', 'w') as file:
#   documents = yaml.dump(yaml_string, file)

# get weights
K.set_learning_phase(0)
weights_baseNet = model.layers[2].get_weights()
# make new session
K.clear_session()
K.set_learning_phase(0)
K.set_learning_phase(False)
bare_model = create_base_network2(input_shape) # create new model
# set weights into new model
bare_model.set_weights(weights_baseNet)
bare_model.trainable = False

# save this graph now
# keep_vars = ['conv1/kernel:0', 'conv1_bn/gamma:0', 'conv1_bn/beta:0', 'conv_dw_1/depthwise_kernel:0', 'conv_dw_1_bn/gamma:0', 'conv_dw_1_bn/beta:0', 'conv_pw_1/kernel:0', 'conv_pw_1_bn/gamma:0', 'conv_pw_1_bn/beta:0', 'conv_dw_2/depthwise_kernel:0', 'conv_dw_2_bn/gamma:0', 'conv_dw_2_bn/beta:0', 'conv_pw_2/kernel:0', 'conv_pw_2_bn/gamma:0', 'conv_pw_2_bn/beta:0', 'conv_dw_3/depthwise_kernel:0', 'conv_dw_3_bn/gamma:0', 'conv_dw_3_bn/beta:0', 'conv_pw_3/kernel:0', 'conv_pw_3_bn/gamma:0', 'conv_pw_3_bn/beta:0', 'conv_dw_4/depthwise_kernel:0', 'conv_dw_4_bn/gamma:0', 'conv_dw_4_bn/beta:0', 'conv_pw_4/kernel:0', 'conv_pw_4_bn/gamma:0', 'conv_pw_4_bn/beta:0', 'conv_dw_5/depthwise_kernel:0', 'conv_dw_5_bn/gamma:0', 'conv_dw_5_bn/beta:0', 'conv_pw_5/kernel:0', 'conv_pw_5_bn/gamma:0', 'conv_pw_5_bn/beta:0', 'conv_dw_6/depthwise_kernel:0', 'conv_dw_6_bn/gamma:0', 'conv_dw_6_bn/beta:0', 'conv_pw_6/kernel:0', 'conv_pw_6_bn/gamma:0', 'conv_pw_6_bn/beta:0', 'conv_dw_7/depthwise_kernel:0', 'conv_dw_7_bn/gamma:0', 'conv_dw_7_bn/beta:0', 'conv_pw_7/kernel:0', 'conv_pw_7_bn/gamma:0', 'conv_pw_7_bn/beta:0', 'conv_dw_8/depthwise_kernel:0', 'conv_dw_8_bn/gamma:0', 'conv_dw_8_bn/beta:0', 'conv_pw_8/kernel:0', 'conv_pw_8_bn/gamma:0', 'conv_pw_8_bn/beta:0', 'conv_dw_9/depthwise_kernel:0', 'conv_dw_9_bn/gamma:0', 'conv_dw_9_bn/beta:0', 'conv_pw_9/kernel:0', 'conv_pw_9_bn/gamma:0', 'conv_pw_9_bn/beta:0', 'conv_dw_10/depthwise_kernel:0', 'conv_dw_10_bn/gamma:0', 'conv_dw_10_bn/beta:0', 'conv_pw_10/kernel:0', 'conv_pw_10_bn/gamma:0', 'conv_pw_10_bn/beta:0', 'conv_dw_11/depthwise_kernel:0', 'conv_dw_11_bn/gamma:0', 'conv_dw_11_bn/beta:0', 'conv_pw_11/kernel:0', 'conv_pw_11_bn/gamma:0', 'conv_pw_11_bn/beta:0', 'conv_dw_12/depthwise_kernel:0', 'conv_dw_12_bn/gamma:0', 'conv_dw_12_bn/beta:0', 'conv_pw_12/kernel:0', 'conv_pw_12_bn/gamma:0', 'conv_pw_12_bn/beta:0', 'conv_dw_13/depthwise_kernel:0', 'conv_dw_13_bn/gamma:0', 'conv_dw_13_bn/beta:0', 'conv_pw_13/kernel:0', 'conv_pw_13_bn/gamma:0', 'conv_pw_13_bn/beta:0', 'dense_1/kernel:0', 'dense_1/bias:0', 'conv1_bn/moving_mean:0', 'conv1_bn/moving_variance:0', 'conv_dw_1_bn/moving_mean:0', 'conv_dw_1_bn/moving_variance:0', 'conv_pw_1_bn/moving_mean:0', 'conv_pw_1_bn/moving_variance:0', 'conv_dw_2_bn/moving_mean:0', 'conv_dw_2_bn/moving_variance:0', 'conv_pw_2_bn/moving_mean:0', 'conv_pw_2_bn/moving_variance:0', 'conv_dw_3_bn/moving_mean:0', 'conv_dw_3_bn/moving_variance:0', 'conv_pw_3_bn/moving_mean:0', 'conv_pw_3_bn/moving_variance:0', 'conv_dw_4_bn/moving_mean:0', 'conv_dw_4_bn/moving_variance:0', 'conv_pw_4_bn/moving_mean:0', 'conv_pw_4_bn/moving_variance:0', 'conv_dw_5_bn/moving_mean:0', 'conv_dw_5_bn/moving_variance:0', 'conv_pw_5_bn/moving_mean:0', 'conv_pw_5_bn/moving_variance:0', 'conv_dw_6_bn/moving_mean:0', 'conv_dw_6_bn/moving_variance:0', 'conv_pw_6_bn/moving_mean:0', 'conv_pw_6_bn/moving_variance:0', 'conv_dw_7_bn/moving_mean:0', 'conv_dw_7_bn/moving_variance:0', 'conv_pw_7_bn/moving_mean:0', 'conv_pw_7_bn/moving_variance:0', 'conv_dw_8_bn/moving_mean:0', 'conv_dw_8_bn/moving_variance:0', 'conv_pw_8_bn/moving_mean:0', 'conv_pw_8_bn/moving_variance:0', 'conv_dw_9_bn/moving_mean:0', 'conv_dw_9_bn/moving_variance:0', 'conv_pw_9_bn/moving_mean:0', 'conv_pw_9_bn/moving_variance:0', 'conv_dw_10_bn/moving_mean:0', 'conv_dw_10_bn/moving_variance:0', 'conv_pw_10_bn/moving_mean:0', 'conv_pw_10_bn/moving_variance:0', 'conv_dw_11_bn/moving_mean:0', 'conv_dw_11_bn/moving_variance:0', 'conv_pw_11_bn/moving_mean:0', 'conv_pw_11_bn/moving_variance:0', 'conv_dw_12_bn/moving_mean:0', 'conv_dw_12_bn/moving_variance:0', 'conv_pw_12_bn/moving_mean:0', 'conv_pw_12_bn/moving_variance:0', 'conv_dw_13_bn/moving_mean:0', 'conv_dw_13_bn/moving_variance:0', 'conv_pw_13_bn/moving_mean:0', 'conv_pw_13_bn/moving_variance:0']

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in bare_model.outputs]) # , keep_var_names=keep_vars
pb_name = tf.train.write_graph(frozen_graph, "exports", "poopyConv.pb", as_text=False)


# lambdaNet = Lambda(euclidean_distance,
#                   output_shape=eucl_dist_output_shape)([processed_a, processed_b])


# load graph again
with tf.gfile.GFile(pb_name, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())


# mkae subgraph --> remove nodes
subgraph = tf.graph_util.extract_sub_graph(graph_def, vars)
tf.train.write_graph(subgraph, "exports", "siamese_2.pb", as_text=False)

