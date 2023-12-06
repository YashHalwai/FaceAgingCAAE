from __future__ import division
import tensorflow as tf
import numpy as np
from PIL import Image

def load_image(image_path, image_size, image_value_range=(-1, 1), is_gray=False):
    """
    Load and preprocess an image from the given path.

    Args:
        image_path (str): Path to the image file.
        image_size (int): Size to which the image should be resized.
        image_value_range (tuple): Expected range of pixel values in the image.
        is_gray (bool): Whether the image should be loaded in grayscale.

    Returns:
        numpy.ndarray: Processed image.
    """
    img = Image.open(image_path)
    if is_gray:
        img = img.convert('L')
    img = img.resize((image_size, image_size), Image.BICUBIC)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0 * (image_value_range[1] - image_value_range[0]) + image_value_range[0]
    return img

def save_batch_images(batch_images, save_path, image_value_range=(-1, 1), size_frame=None):
    """
    Save a batch of images in a grid.

    Args:
        batch_images (numpy.ndarray): Batch of images to save.
        save_path (str): Path to save the grid image.
        image_value_range (tuple): Expected range of pixel values in the image.
        size_frame (list): Number of rows and columns in the grid.
    """
    num_images, height, width, num_channels = batch_images.shape

    if size_frame is None:
        size_frame = [int(np.sqrt(num_images))] * 2

    # Ensure the number of images matches the size_frame
    num_images = min(num_images, size_frame[0] * size_frame[1])

    # Create the grid
    grid = np.zeros((height * size_frame[0], width * size_frame[1], num_channels), dtype=np.uint8)

    for idx in range(num_images):
        i = idx % size_frame[0]
        j = idx // size_frame[0]
        grid[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = (
            batch_images[idx, :, :, :] * 255.0 / (image_value_range[1] - image_value_range[0]) +
            (image_value_range[1] * image_value_range[0]) / (image_value_range[1] - image_value_range[0])
        ).astype(np.uint8)

    # Save the grid image
    imsave(save_path, grid)

def conv2d(input_map, num_output_channels, size_kernel=5, name='conv2d'):
    """
    Perform 2D convolution.

    Args:
        input_map (tf.Tensor): Input tensor.
        num_output_channels (int): Number of output channels.
        size_kernel (int): Size of the convolution kernel.
        name (str): Name of the operation.

    Returns:
        tf.Tensor: Output tensor.
    """
    with tf.variable_scope(name):
        input_shape = input_map.get_shape().as_list()
        filter_shape = [size_kernel, size_kernel, input_shape[-1], num_output_channels]

        # Initialize weights with He initialization
        weights = tf.get_variable(
            name='weights',
            shape=filter_shape,
            initializer=tf.initializers.he_normal(),
        )

        # Perform convolution
        output_map = tf.nn.conv2d(input_map, weights, strides=[1, 2, 2, 1], padding='SAME')

    return output_map

def deconv2d(input_map, output_shape, size_kernel=5, stride=2, name='deconv2d'):
    """
    Perform 2D deconvolution.

    Args:
        input_map (tf.Tensor): Input tensor.
        output_shape (list): Shape of the output tensor.
        size_kernel (int): Size of the deconvolution kernel.
        stride (int): Stride of the deconvolution.
        name (str): Name of the operation.

    Returns:
        tf.Tensor: Output tensor.
    """
    with tf.variable_scope(name):
        input_shape = input_map.get_shape().as_list()
        filter_shape = [size_kernel, size_kernel, output_shape[-1], input_shape[-1]]

        # Initialize weights with He initialization
        weights = tf.get_variable(
            name='weights',
            shape=filter_shape,
            initializer=tf.initializers.he_normal(),
        )

        # Perform deconvolution
        output_map = tf.nn.conv2d_transpose(input_map, weights, output_shape, strides=[1, stride, stride, 1])

    return output_map

def fc(input_vector, num_output_length, name='fc'):
    """
    Perform fully connected layer.

    Args:
        input_vector (tf.Tensor): Input tensor.
        num_output_length (int): Length of the output tensor.
        name (str): Name of the operation.

    Returns:
        tf.Tensor: Output tensor.
    """
    with tf.variable_scope(name):
        input_shape = input_vector.get_shape().as_list()

        # Flatten the input vector if needed
        if len(input_shape) > 2:
            input_vector = tf.reshape(input_vector, [input_shape[0], -1])

        # Initialize weights with He initialization
        weights = tf.get_variable(
            name='weights',
            shape=[input_vector.get_shape().as_list()[1], num_output_length],
            initializer=tf.initializers.he_normal(),
        )

        # Perform matrix multiplication
        output_vector = tf.matmul(input_vector, weights)

    return output_vector

def lrelu(x, leak=0.2):
    """
    Apply leaky ReLU activation.

    Args:
        x (tf.Tensor): Input tensor.
        leak (float): Leaking factor for negative values.

    Returns:
        tf.Tensor: Output tensor.
    """
    return tf.maximum(x, leak * x)





def concat_label(x, y, duplicate=True):
    if x is None or y is None:
        raise ValueError("Input tensors cannot be None.")

    print(f"x shape: {x.shape}, y shape: {y.shape}")

    x = tf.debugging.check_numerics(x, "x contains NaN or Inf values.")
    y = tf.debugging.check_numerics(y, "y contains NaN or Inf values.")

    if duplicate:
        y = tf.tile(y, [1, 1, 1, tf.shape(x)[3] // tf.shape(y)[3]])
        
    # print(f"x: {x}")
    # print(f"y: {y}")

    return tf.concat([x, y], axis=3)

    





def instance_norm(x, name='instance_norm'):
    """
    Apply instance normalization.

    Args:
        x (tf.Tensor): Input tensor.
        name (str): Name of the operation.

    Returns:
        tf.Tensor: Output tensor.
    """
    with tf.variable_scope(name):
        depth = x.get_shape()[3]
        scale = tf.get_variable('scale', [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable('offset', [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset
