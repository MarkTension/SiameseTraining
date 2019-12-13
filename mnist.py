from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf


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

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# name="mnistkeras1"
# model.save(name + ".h5")
# tf.keras.models.save_model(name + ".h5", save_format="tf")
# model_saved = keras.models.load_model(name + ".h5") #, custom_objects={'contrastive_loss': contrastive_loss}
# keras.experimental.export_saved_model(model_saved, 'exports/' + name + ".pb")


frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

pb_name = tf.train.write_graph(frozen_graph, "exports", "siamese1.pb", as_text=False)

# load graph again
with tf.gfile.GFile(pb_name, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

# mkae subgraph --> remove nodes
subgraph = tf.graph_util.extract_sub_graph(graph_def, ['dense_2/Softmax', 'conv2d_1/kernel', 'conv2d_1/bias', 'conv2d_2/kernel', 'conv2d_2/bias', 'dense_1/kernel', 'dense_1/bias', 'dense_2/kernel', 'dense_2/bias'])
tf.train.write_graph(subgraph, "exports", "my_model2.pb", as_text=False)
print('poopy')

