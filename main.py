# imports
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np


def create_model():
    """
    function for creating the model of the neural network using Tensorflow and Keras
    Returns a model to be used
    """
    model = tf.keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model


def encode(model: tf.keras.Model, datapath: str) -> ():
    """
    :param model: Takes an input model and performs the encoding step for separation with
     input files
    :param datapath: Takes input data path for files to encode
    :return:
    """


def trainSeperation(model: tf.keras.Model, path: str,
                    batch_size: int
                    ) -> ():
    """
    Starts training and saves checkpoints
    :param path: path to save checkpoints
    :param batch_size: epoch amount
    :param model: Input model to train separation
    :return:
    """
    model.save_weights(path.format(epoch=0))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * batch_size
    )



def continueTraining(model: tf.keras.Model, checkpoint,
                     batch_size:int):
    """
    Continues training the model based on checkpoints
    :param batch_size: epoch amount
    :param model:
    :param checkpoint:
    :return:
    """


if __name__ == "__main__":
    batch_size = 6
    model = create_model()
    checkpoint_directory = "checkpoints"
    if not os.listdir(checkpoint_directory):
        trainSeperation(model, checkpoint_directory, batch_size)
    else:
        continueTraining(model,
                         os.listdir(checkpoint_directory[-1]),
                         batch_size)
    model.summary()