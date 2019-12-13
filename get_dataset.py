import numpy as np
from keras.datasets import mnist, fashion_mnist
from PIL import Image
import os.path


def resize(data_in, imsize):
  # make  [60000, 96, 96, 3] array
  data_out = Image.fromarray(data_in[0].astype('uint8'))
  data_out = np.asarray(data_out.resize((imsize, imsize)))
  data_out = np.expand_dims(data_out, 0)
  data_out = np.expand_dims(data_out, 3)
  data_out = np.repeat(data_out, repeats=3, axis=3)
  data_out = np.repeat(data_out, repeats=data_in.shape[0], axis=0)

  # make iterator
  image_old = (sample for sample in data_in)

  # replace array with dataset
  for i in range(data_in.shape[0]):
    # rescale all images to 96 * 96
    image = Image.fromarray(image_old.__next__().astype('uint8'))
    image = np.asarray(image.resize((imsize, imsize)))
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 3)
    image = np.repeat(image, repeats=3, axis=3)

    data_out[i] = image
    if i % 900 == 0:
      print(f"image resize step {i} of {data_out.shape[0]}")

  return data_out


def get_fashion_mnist(imsize):

  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  xtrn = resize(x_train, imsize)
  xtst = resize(x_test, imsize)


  # normalize to 0 ~ 1
  xtrn = xtrn.astype('float32')
  xtst = xtst.astype('float32')
  xtrn /= 255
  xtst /= 255

  return (xtrn, y_train), (xtst, y_test)