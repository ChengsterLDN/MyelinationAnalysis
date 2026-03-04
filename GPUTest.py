import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())  # lists ALL devices, CPU and GPU
print(tf.test.is_built_with_cuda())       # True means TF was compiled with GPU support