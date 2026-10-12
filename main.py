# @title 2.1. Importeer modules
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical

def train_model(input_size, hidden_size, output_size, x_train, y_train):
  x_train = x_train.astype('float32') / 255
  x_train = x_train.reshape(-1, input_size)
  y_train = to_categorical(y_train, output_size)

  model = Sequential([
      Input(shape=(input_size,)),
      Dense(hidden_size, activation='relu'),
      Dense(output_size, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train, epochs=1, batch_size=32)

  return model

def test_model(model, input_size, output_size, x_test, y_test):
  x_test = x_test.astype('float32') / 255
  x_test_flat = x_test.reshape(-1, input_size)
  y_test = to_categorical(y_test, output_size)

  test_loss, test_acc = model.evaluate(x_test_flat, y_test)
  print()
  print(f'Modelprestatie: {test_acc:.4f}')

  predictions = model.predict(x_test_flat)

  print()
  print(f'3 verkeerde voorspellingen:')
  count = 0
  for i in range(len(y_test)):
    if count >= 3:
      break

    if np.argmax(y_test[i]) != np.argmax(predictions[i]):
      count += 1
      plt.imshow(x_test[i], cmap='gray')
      plt.title(f'Voorspeld: {np.argmax(predictions[i])}, correct: {np.argmax(y_test[i])}')
      plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

mat = x_train[4]
col_maxes = [max([len(("{:g}").format(x)) for x in col]) for col in mat.T]
for x in mat:
    for i, y in enumerate(x):
        print(("{:3}").format(y), end="  ")
    print("")