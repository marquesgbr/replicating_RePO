"""Load models saved by both legacy TensorFlow and Keras 3."""

import os

import tensorflow as tf


class SavedModelAdapter:
    """Expose a single-output SavedModel with the Keras call interface."""

    def __init__(self, path):
        self._saved_model = tf.saved_model.load(path)
        signatures = self._saved_model.signatures
        if "serving_default" not in signatures:
            raise ValueError(
                f"SavedModel at {path!r} has no 'serving_default' signature."
            )

        self._serve = signatures["serving_default"]
        _, keyword_inputs = self._serve.structured_input_signature
        if len(keyword_inputs) != 1:
            raise ValueError(
                f"Expected one input in 'serving_default', found {len(keyword_inputs)}."
            )
        self._input_name = next(iter(keyword_inputs))

    def __call__(self, inputs, training=False):
        del training
        outputs = self._serve(**{self._input_name: inputs})
        if len(outputs) != 1:
            raise ValueError(
                f"Expected one output in 'serving_default', found {len(outputs)}."
            )
        return next(iter(outputs.values()))


def load_model(path):
    """Load a Keras model or a legacy TensorFlow SavedModel."""
    if os.path.isdir(path):
        return SavedModelAdapter(path)
    return tf.keras.models.load_model(path)
