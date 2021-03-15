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
    """Returns a model that transforms audio clips into feature vectors.

    The resulting model takes a batch of 1D audio clips and produces
    a batch of sequences of feature vectors. Each sequence is the same length
    as the input clips. During inference, either the final feature vectors or
    the mean features over each sequence or a more sophisticated limit
    approximation may be used as the fingerprint.

    :param int: chunk_size: number of samples in FFT chunks
    :return: TensorFlow Keras model

    Note: LSTM is being used without the Bidirectional wrapper to entertain
    the idea that the model may be used for real-time speech analysis. Were it
    bidirectional, a full clip of audio would be necessary to run the model.
    Since it is not bidirectional, it is possible to run the model on a live
    stream of audio.
    """
    model = tf.keras.Sequential()

    model.add(kl.Reshape([-1, chunk_size]))
    model.add(RFFT())
    model.add(kl.Conv1D(chunk_size, kernel_size=1,
                        activation=kl.LeakyReLU()))
    model.add(kl.Conv1D(chunk_size // 2, kernel_size=1,
                        activation=kl.LeakyReLU()))
    model.add(kl.LSTM(units=128,
                      return_sequences=True))
    model.add(kl.LSTM(units=128,
                      return_sequences=True))
    model.add(kl.Conv1D(64, kernel_size=1,
                        activation=kl.LeakyReLU()))

    return model


def make_inference_model(base_model):
    model = tf.keras.Sequential()

    model.add(base_model)
    model.add(kl.GlobalAveragePooling1D())

    return model
