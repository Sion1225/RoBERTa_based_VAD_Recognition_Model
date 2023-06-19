# Part of Define and Build FFNN_VAD_Model

import tensorflow as tf

# Build model
def FFNN_VAD_model(units, kernel_l2_lambda, activity_l2_lambda, dropout_late):
    inputs = tf.keras.layers.Input(shape=(3,))
    hidden = tf.keras.layers.Dense(
        units=units,
        kernel_regularizer=tf.keras.regularizers.L2(kernel_l2_lambda), 
        activity_regularizer=tf.keras.regularizers.L2(activity_l2_lambda),
        activation="gelu",
        kernel_initializer = "he_uniform" #he_normal or he_uniform
    )(inputs)
    hidden = tf.keras.layers.Dropout(dropout_late)(hidden)
    outputs = tf.keras.layers.Dense(3,activation="linear")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model