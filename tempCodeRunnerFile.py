 self.input_image = tf.placeholder(
            tf.float32,
            [None, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )