"""Three TensorFlow image models for 13,065-way character recognition.

All models share the same input size, loss, and optimizer. Only the trunk
changes, so the test-set comparison is about architecture.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


MODEL_NAMES = ("shallow_cnn", "deep_cnn", "residual_cnn")


def _conv_bn_relu(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def _residual_block(x, filters: int, stride: int):
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    in_channels = x.shape[-1]
    if stride != 1 or in_channels != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(x)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([y, shortcut])
    return layers.ReLU()(y)


def shallow_cnn(num_classes: int, img_size: int) -> tf.keras.Model:
    """Two conv stages, then a dense layer. Smallest of the three models."""
    inputs = layers.Input(shape=(img_size, img_size, 1), name="image")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="character")(x)
    return models.Model(inputs, outputs, name="shallow_cnn")


def deep_cnn(num_classes: int, img_size: int) -> tf.keras.Model:
    """Four conv-BN-ReLU blocks with global average pooling before the classifier."""
    inputs = layers.Input(shape=(img_size, img_size, 1), name="image")
    x = inputs
    for filters in (32, 64, 128, 256):
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="character")(x)
    return models.Model(inputs, outputs, name="deep_cnn")


def residual_cnn(num_classes: int, img_size: int) -> tf.keras.Model:
    """A small residual network. Shortcut paths keep the deeper stack trainable."""
    inputs = layers.Input(shape=(img_size, img_size, 1), name="image")
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = _residual_block(x, 32, stride=1)
    x = _residual_block(x, 64, stride=2)
    x = _residual_block(x, 128, stride=2)
    x = _residual_block(x, 128, stride=1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="character")(x)
    return models.Model(inputs, outputs, name="residual_cnn")


BUILDERS = {
    "shallow_cnn": shallow_cnn,
    "deep_cnn": deep_cnn,
    "residual_cnn": residual_cnn,
}


def build_model(name: str, num_classes: int, img_size: int, learning_rate: float) -> tf.keras.Model:
    """Create one compiled model. The softmax width is the number of characters."""
    if name not in BUILDERS:
        known = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model '{name}'. Choose from: {known}")
    model = BUILDERS[name](num_classes, img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )
    return model


def model_description(name: str) -> str:
    """Short architecture note stored next to the accuracy table."""
    notes = {
        "shallow_cnn": "Conv32 x2, pool, Conv64 x2, pool, Dense256, softmax",
        "deep_cnn": "Four Conv-BN blocks (32/64/128/256), GAP, Dense512, softmax",
        "residual_cnn": "Stem32 + residual blocks 32/64/128/128, GAP, softmax",
    }
    return notes[name]
