import tensorflow as tf

from models import make_base_model


def hypersphere_loss(x):
    """Loss for constraining the vector x to the unit hypersphere.
    """
    magnitude = tf.reduce_sum(tf.square(x), axis=-1)
    return tf.maximum(1. / (magnitude + 1e-12),
                      tf.exp(magnitude - 1.))


def train_step(base_model, anchor, positive, negative, optimizer):
    """Train using matching/non-matching triplets.

    Based on [FaceNet](http://arxiv.org/abs/1503.03832).
    """
    # Only calculate losses for negative if it will make a difference
    use_negative = negative is not None

    with tf.GradientTape() as tape:
        a_feat = base_model(anchor)
        p_feat = base_model(positive)
        if use_negative:
            n_feat = base_model(negative)

        loss = 0.

        # Constrain features to hypersphere
        loss += tf.reduce_mean(hypersphere_loss(a_feat))
        loss += tf.reduce_mean(hypersphere_loss(p_feat))
        if use_negative:
            loss += tf.reduce_mean(hypersphere_loss(n_feat))

        # Apply anchor-positive loss
        p_dist = tf.reduce_sum(tf.square(a_feat - p_feat), axis=-1)
        loss += tf.reduce_mean(p_dist)
        if use_negative:
            # Apply anchor-negative loss
            n_dist = tf.reduce_sum(tf.square(a_feat - n_feat), axis=-1)
            loss += -tf.reduce_mean(n_dist)

        # Minimize prediction time-variance
        loss += tf.reduce_mean(tf.square(
            a_feat - tf.reduce_mean(a_feat, axis=1)))
        loss += tf.reduce_mean(tf.square(
            p_feat - tf.reduce_mean(p_feat, axis=1)))
        if use_negative:
            loss += tf.reduce_mean(tf.square(
                n_feat - tf.reduce_mean(n_feat, axis=1)))

    grads = tape.gradient(loss, base_model.trainable_variables)
    grads_and_vars = zip(grads, base_model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars)


if __name__ == '__main__':
    model = make_base_model(1024)
