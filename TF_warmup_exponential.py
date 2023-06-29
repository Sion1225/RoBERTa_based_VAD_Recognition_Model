import tensorflow as tf
import numpy as np

class TF_warmup_exponential(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, max_lr: float, min_lr: float, num_warmup: int, a: float):
        super(TF_warmup_exponential, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.num_warmup = num_warmup
        self.a = a

    def __call__(self, step: int) -> tf.Tensor :
        return tf.cond(
            step <= self.num_warmup,
            lambda: self.max_lr / self.num_warmup * step,
            lambda: self.max_lr * np.exp(-self.a * (step - self.num_warmup)) + self.min_lr
        )

    def get_config(self) -> dict:
        return {
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "num_warmup": self.num_warmup,
            "a": self.a
        }