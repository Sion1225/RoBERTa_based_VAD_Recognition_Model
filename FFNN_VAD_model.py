"""Part of Define and Build FFNN_VAD_Model."""

import tensorflow as tf

# Build model
class FFNN_VAD_model(tf.keras.Model):
    def __init__(self, units: int, kernel_l2_lambda: float, activity_l2_lambda: float, dropout_rate: float):
        super(FFNN_VAD_model, self).__init__()

        self.units = units
        self.kernel_l2_lambda = kernel_l2_lambda
        self.activity_l2_lambda = activity_l2_lambda
        self.dropout_rate = dropout_rate

        self.dense1 = tf.keras.layers.Dense(
            units=self.units,
            kernel_regularizer=tf.keras.regularizers.L2(self.kernel_l2_lambda),
            activity_regularizer=tf.keras.regularizers.L2(self.activity_l2_lambda),
            activation="gelu",
            kernel_initializer="he_normal"  # he_normal or he_uniform
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.output_layer = tf.keras.layers.Dense(3, activation="linear")

    def call(self, inputs):
        hidden = self.dense1(inputs)
        hidden = self.dropout(hidden)
        ouputs = self.output_layer(hidden)
        return ouputs

    def get_config(self):
        config = super(FFNN_VAD_model, self).get_config()
        config.update({
            'units': self.units,
            'kernel_l2_lambda': self.kernel_l2_lambda,
            'activity_l2_lambda': self.activity_l2_lambda,
            'dropout_rate': self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)