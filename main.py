# day-1 simple neural network building and analysis of loss and optimizer
# import libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

# simple neural network and then we have to create loss and optimizer function
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# loss function may guess answers which could be wrong but optimizer function will reduce the errors
model.compile(optimizer='sgd', loss='mean_squared_error')

# providing data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# training the model
model.fit(xs, ys, epochs=500)

# prediction
print(model.predict([10.0]))