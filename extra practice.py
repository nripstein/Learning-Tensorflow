import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

# define data
# X = np.arange(0, 1000, 5)
# y = X + 100
X = np.linspace(0, 100, 10)
y = X + 10
# print(len(X))
# print(0.9*len(X))
# split into test and train
X_train, X_test = X[:int(0.9*len(X))], X[int(0.9*len(X)):]
y_train, y_test = y[:int(0.9*len(X))], y[int(0.9*len(X)):]


# create model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(5, activation="relu"),
    # tf.keras.layers.Dense(100, name="deep_layer2", activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])
#
# # compile model
model.compile(loss=tf.keras.losses.mse,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["mse"])
#
# # fit model
model.fit(X_train, y_train, epochs=50, verbose=1, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="blue", label="Training Data")
    plt.scatter(test_data, test_labels, c="green", label="Testing Data")
    plt.scatter(test_data, predictions, c="red", label="Predictions")
    plt.legend()
    plt.show()

plot_predictions(X_train, y_train, X_test, y_test, y_pred)



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
model = LinearRegression()

# fit the model to the training data
model.fit(X_train.reshape(-1, 1), y_train)

# make predictions on the test data
y_pred = model.predict(X_test.reshape(-1, 1))

# evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

