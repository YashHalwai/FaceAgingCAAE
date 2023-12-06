# utils.py
import tensorflow as tf

def show_all_variables():
    for var in tf.trainable_variables():
        print(var.name)
