# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to convert tf functions and keras models using the Python API.
It also demonstrates converting saved_models from the command line.
"""

import tensorflow as tf
import tf2onnx
import numpy as np
import onnxruntime as ort
import os


def create_keras_model():
    keras_model = tf.keras.Sequential()
    keras_model.add(tf.keras.layers.Dense(4, activation="relu"))

    input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)

    x_val = np.ones((3, 3), np.float32)

    print("Keras result")
    print(keras_model(x_val).numpy())

    print("ORT result")
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    res = sess.run(None, {'x': x_val})
    print(res[0])

    return keras_model


def save_model(model):
    model.save("tf-onnx-test-model")
    os.system("python3 -m tf2onnx.convert --saved-model tf-onnx-test-model --output test-model.onnx --opset 13")


def onnx_inference():
    dense_input = np.ones((3, 3), np.float32)
    # dense_input_tf = tf.constant([3, 3])
    print("ORT result")
    sess = ort.InferenceSession("../../models/test-model.onnx")
    res = sess.run(None, {'dense_input': dense_input})
    print(res[0])

    print("Conversion succeeded")


if __name__ == '__main__':
    # model = create_keras_model()
    # save_model(model)
    onnx_inference()
