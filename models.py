import tensorflow as tf
import tensorflow.keras.layers as kl


class RFFT(tf.keras.layers.Layer):
    """Keras wrapper for tf.signal.rfft with length normalization.
    """

    def __init__(self):
        super(RFFT, self).__init__()

    def call(self, inputs):
        norm_divisor = tf.cast(inputs.get_shape().as_list()[-1], 'float32')
        return tf.abs(tf.signal.rfft(inputs)) / norm_divisor


def make_base_model(chunk_size):
    model = tf.keras.Sequential()

    model.add(kl.Reshape([-1, chunk_size]))
    model.add(RFFT())
    model.add(kl.Conv1D(chunk_size, kernel_size=1,
                        activation=kl.LeakyReLU()))
    model.add(kl.Conv1D(chunk_size // 2, kernel_size=1,
                        activation=kl.LeakyReLU()))
    model.add(kl.Bidirectional(kl.LSTM(units=128,
                                       return_sequences=True)))
    model.add(kl.Bidirectional(kl.LSTM(units=128,
                                       return_sequences=True)))
    model.add(kl.Conv1D(64, kernel_size=1,
                        activation=kl.LeakyReLU()))

    return model


def make_inference_model(base_model):
    model = tf.keras.Sequential()

    model.add(base_model)
    model.add(kl.GlobalAveragePooling1D())

    return model
