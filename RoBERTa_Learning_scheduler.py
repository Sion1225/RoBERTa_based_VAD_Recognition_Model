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

    def __call__(self, step_now):
        if step_now < self.num_warmup:
            return (float(step_now) / float(max(1, self.num_warmup))) * self.max_lr
        
        return max(0.0, self.max_lr * ((float(self.num_training - step_now) / float(max(1, self.num_training - self.num_warmup)))))
    
    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps
            }