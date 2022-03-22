import pandas as pd
import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from keras.optimizers import gradient_descent_v2 as SGD
import keras
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"moore.csv").values
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]

y = np.log(y)
X = X - X.mean()

model = keras.Sequential()
model.add(Input(shape=(1,)))
model.add(Dense(1))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mse')


def schedule(epochs, lr):
    if epochs >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)
r = model.fit(X, y, epochs=200, callbacks=scheduler)

plt.plot(r.history['loss'], label='loss')
plt.show()

a = model.layers[0].get_weights()[0][0, 0]

print("Time to double: ", np.log(2)/a)

# Analytical solution
X = np.array(X).flatten()
y = np.array(y)
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(y) - y.mean()*X.sum())/denominator
b = (y.mean()*X.dot(X) - X.mean()*X.dot(y)) / denominator
print(a, b)
print("Time to double: ", np.log(2)/a)








