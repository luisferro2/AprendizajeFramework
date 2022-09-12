""" This file contains the implementation of a simple neural net to 
predict the outcome of the survival of people aboard the Titanic using
the framework Tensorflow.

Author: Luis Ignacio Ferro Salinas A01378248
Last update: september 10th, 2022

Please close the matplotlib windows as they appear so the program can 
flow completely.
"""

from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC
import matplotlib.pyplot as plt

# Simple preprocessing to the Titanic dataset.
titanic_df = pd.read_csv("train-2.csv")
titanic_df.info()
titanic_df.drop("Cabin", axis=1, inplace=True)
titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)
titanic_df.dropna(subset=["Embarked"], inplace=True)
titanic_df.drop("Name", axis=1, inplace=True)
titanic_df = pd.get_dummies(titanic_df, columns=["Sex"])
titanic_df.drop("Ticket", axis=1, inplace=True)
titanic_df.Embarked.replace({"S": 1, "C": 2, "Q": 3}, inplace=True)

# The target variable and the independent variables.
X = titanic_df.drop("Survived", axis=1)
y = titanic_df["Survived"]

# Since train_split_test has shuffle=True, every time I call it,
# the training and testing datasets are different.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    shuffle=True)

# My model receives 9 features and outputs one probability in the
# [0, 1] range, representing whether or not the passenger survived
# the Titanic crash or not.

# my_tf_model = Sequential(layers=[Input(9), Dense(1, activation="sigmoid")])# one neuron
# my_tf_model = Sequential(layers=[Input(9), Dense(32, activation="sigmoid"), Dense(8, activation="sigmoid"), Dense(1, activation="sigmoid")])# 2 hidden layers
my_tf_model = Sequential(layers=[Input(9),
                                 Dense(16, activation="relu"),
                                 Dense(1, activation="sigmoid")])

print("Mi modelo tiene la siguientes características",
      my_tf_model.summary())

# I was very intrigued because this framework has a lot of the metrics
# we saw in class like, AUC, accuracy, even TP, FP, FP, FN.
my_tf_model.compile(optimizer="adam", loss="binary_crossentropy",
                    metrics=[BinaryAccuracy(), AUC()])

history = my_tf_model.fit(X_train, y_train, verbose=1,
                          batch_size=256, epochs=300,
                          validation_split=0.1)

# I plot the metrics and loss values for both the training and
# validation so I can visualize the progress of my model with
# different configurations.
history_dictionary = history.history
fig = plt.figure(figsize=(20, 10))
for i, indicator in enumerate(history_dictionary):
    if indicator[0:3] == "val":
        break
    fig.add_subplot(2, 2, i + 1)
    plt.plot(range(len(history_dictionary[indicator])),
             history_dictionary[indicator], label="training")
    plt.plot(range(len(history_dictionary["val_" + indicator])),
             history_dictionary["val_" + indicator], label="validation")
    plt.legend()
    plt.title(indicator)
plt.show()

# Here I predict the outcome for the test data I set aside previously
# and plot the predictions against the truth to visualize the accuracy
# of the model.
y_pred = my_tf_model.predict(X_test)

plt.figure(figsize=(28, 10))
plt.plot(range(len(y_pred)), y_pred, label="predictions")
plt.plot(range(len(y_test)), y_test, label="true")
plt.xlabel("sample number")
plt.ylabel("Survived")
plt.title("Predictions analysis")
plt.legend()
plt.show()
