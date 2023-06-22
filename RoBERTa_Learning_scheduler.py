"""
Linear schedule with warmup.
This is implementation of the Learning scheduler of the RoBERTa paper for tensorflow.
RoBERTa: A Robustly Optimized BERT Pretraining Approach (Yinhan Liu, Myle Ott et al., 2019)
"""

import tensorflow as tf

class Linear_schedule_with_warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr: int, num_warmup: int , num_traning: int):
        self.max_lr = max_lr
        self.num_warmup = num_warmup
        self.num_training = num_traning

    def __call__(self, step_now) -> float:
        return tf.cond(
            step_now < self.num_warmup,
            lambda: (tf.cast(step_now, tf.float32) / tf.cast(tf.maximum(1.0, tf.cast(self.num_warmup, tf.float32)), tf.float32)) * self.max_lr,
            lambda: tf.maximum(0.0, self.max_lr * ((tf.cast(self.num_training, tf.float32) - tf.cast(step_now, tf.float32)) / tf.cast(tf.maximum(1.0, tf.cast(self.num_training - self.num_warmup, tf.float32)), tf.float32)))
            )
    
    def get_config(self) -> dict:
        return {
            "max_lr": self.max_lr,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps
            }