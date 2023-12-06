# This code is an alternative implementation of the paper by
# Zhifei Zhang, Yang Song, and Hairong Qi. "Age Progression/Regression by Conditional Adversarial Autoencoder."
# IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
#
# Date:     Mar. 24th, 2017
#
# Please cite above paper if you use this code
#

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from ops import conv2d, fc, deconv2d, lrelu, concat_label, load_image, save_batch_images



# ...

class FaceAging(object):
    def __init__(self,
                 session,
                 size_image=128,
                 size_kernel=5,
                 size_batch=100,
                 num_input_channels=3,
                 num_encoder_channels=64,
                 num_z_channels=50,
                 num_categories=10,
                 num_gen_channels=1024,
                 enable_tile_label=True,
                 tile_ratio=1.0,
                 is_training=True,
                 save_dir='./save',
                 dataset_name='UTKFace'
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.dataset_name = dataset_name

        # ************************************* input to graph ********************************************************
        self.input_image = tf.keras.Input(
    shape=(self.size_image, self.size_image, self.num_input_channels),
    name='input_images',
    dtype=tf.float32
)


        self.gender = tf.Variable(tf.zeros([self.size_batch, 2], dtype=tf.float32), name='gender_labels')
        self.z_prior = tf.Variable(tf.zeros([self.size_batch, self.num_z_channels], dtype=tf.float32), name='z_prior')

        # Placeholder for age labels
        self.age = tf.keras.Input(
            shape=(num_categories,),  # Assuming one-hot encoding for age categories
            name='age',
            dtype=tf.float32
        )

        # Placeholder for z
        self.z = self.encoder(
            image=self.input_image
        )

        self.G = self.generator(
    self.input_z, self.input_age, self.options, reuse=False, name="generatorA2B"
)


        # Rest of the __init__ method remains unchanged...

# ...



        # discriminator on z
        self.D_z, self.D_z_logits = self.discriminator_z(
            z=self.z,
            is_training=self.is_training
        )


        # discriminator on G
        self.D_G, self.D_G_logits = self.discriminator_img(
            image=self.G,
            y=self.age,
            gender=self.gender,
            is_training=self.is_training
        )


        # discriminator on z_prior
        self.D_z_prior, self.D_z_prior_logits = self.discriminator_z(
            z=self.z_prior,
            is_training=self.is_training,
            reuse_variables=True
        )

        # discriminator on input image
        self.D_input, self.D_input_logits = self.discriminator_img(
            image=self.input_image,
            y=self.age,
            gender=self.gender,
            is_training=self.is_training,
            reuse_variables=True
        )


        # ************************************* loss functions *******************************************************
        # loss function of encoder + generator
        # self.EG_loss = tf.nn.l2_loss(self.input_image - self.G) / self.size_batch  # L2 loss
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss

        # loss function of discriminator on z
        self.D_z_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_prior_logits, labels=tf.ones_like(self.D_z_prior_logits))
        )
        self.D_z_loss_z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.zeros_like(self.D_z_logits))
        )
        self.E_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_z_logits, labels=tf.ones_like(self.D_z_logits))
        )
        # loss function of discriminator on image
        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_input_logits, labels=tf.ones_like(self.D_input_logits))
        )
        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.zeros_like(self.D_G_logits))
        )
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G_logits, labels=tf.ones_like(self.D_G_logits))
        )

        # total variation to smooth the generated image
        tv_y_size = self.size_image
        tv_x_size = self.size_image
        self.tv_loss = (
            (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
            (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1, :]) / tv_x_size)) / self.size_batch

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on z
        self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
        # variables of discriminator on image
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]


        # ************************************* collect the summary ***************************************
        self.z_summary = tf.summary.histogram('z', self.z)
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.D_z_loss_z_summary = tf.summary.scalar('D_z_loss_z', self.D_z_loss_z)
        self.D_z_loss_prior_summary = tf.summary.scalar('D_z_loss_prior', self.D_z_loss_prior)
        self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
        self.D_z_logits_summary = tf.summary.histogram('D_z_logits', self.D_z_logits)
        self.D_z_prior_logits_summary = tf.summary.histogram('D_z_prior_logits', self.D_z_prior_logits)
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=2)


    def train(self,
              num_epochs=200,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # use the saved checkpoint to initialize the network
              use_init_model=True,  # use the init model to initialize the network
              weigts=(0.0001, 0, 0)  # the weights of adversarial loss and TV loss
              ):

        # *************************** load file names of images ******************************************************
        file_names = glob(os.path.join('./data', self.dataset_name, '*.jpg'))
        size_data = len(file_names)
        np.random.seed(seed=2017)
        if enable_shuffle:
            np.random.shuffle(file_names)

        # *********************************** optimizer **************************************************************
        # over all, there are three loss functions, weights may differ from the paper because of different datasets
        self.loss_EG = self.EG_loss + weigts[0] * self.G_img_loss + weigts[1] * self.E_z_loss + weigts[2] * self.tv_loss # slightly increase the params
        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )


        # optimizer for encoder + generator
        with tf.variable_scope('opt', reuse=tf.AUTO_REUSE):
            self.EG_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_EG,
                global_step=self.EG_global_step,
                var_list=self.E_variables + self.G_variables
            )


            # optimizer for discriminator on z
            self.D_z_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Dz,
                var_list=self.D_z_variables
            )

            # optimizer for discriminator on image
            self.D_img_optimizer = tf.train.AdamOptimizer(
                learning_rate=EG_learning_rate,
                beta1=beta1
            ).minimize(
                loss=self.loss_Di,
                var_list=self.D_img_variables
            )


        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.z_summary, self.z_prior_summary,
            self.D_z_loss_z_summary, self.D_z_loss_prior_summary,
            self.D_z_logits_summary, self.D_z_prior_logits_summary,
            self.EG_loss_summary, self.E_z_loss_summary,
            self.D_img_loss_input_summary, self.D_img_loss_G_summary,
            self.G_img_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)


        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_files = file_names[0:self.size_batch]
        file_names[0:self.size_batch] = []
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)
        sample_label_age = np.ones(
            shape=(len(sample_files), self.num_categories),
            dtype=np.float32
        ) * self.image_value_range[0]
        sample_label_gender = np.ones(
            shape=(len(sample_files), 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i, label in enumerate(sample_files):
            label = int(str(sample_files[i]).split('/')[-1].split('_')[0])
            if 0 <= label <= 5:
                label = 0
            elif 6 <= label <= 10:
                label = 1
            elif 11 <= label <= 15:
                label = 2
            elif 16 <= label <= 20:
                label = 3
            elif 21 <= label <= 30:
                label = 4
            elif 31 <= label <= 40:
                label = 5
            elif 41 <= label <= 50:
                label = 6
            elif 51 <= label <= 60:
                label = 7
            elif 61 <= label <= 70:
                label = 8
            else:
                label = 9
            sample_label_age[i, label] = self.image_value_range[-1]
            gender = int(str(sample_files[i]).split('/')[-1].split('_')[1])
            sample_label_gender[i, gender] = self.image_value_range[-1]

        # ******************************************* training *******************************************************
        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")
                # load init model
                if use_init_model:
                    if not os.path.exists('init_model/model-init.data-00000-of-00001'):
                        from init_model.zip_opt import join
                        try:
                            join('init_model/model_parts', 'init_model/model-init.data-00000-of-00001')
                        except:
                            raise Exception('Error joining files')
                    self.load_checkpoint(model_path='init_model')


        # epoch iteration
        num_batches = len(file_names) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch images and labels
                batch_files = file_names[ind_batch*self.size_batch:(ind_batch+1)*self.size_batch]
                batch = [load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_label_age = np.ones(
                    shape=(len(batch_files), self.num_categories),
                    dtype=np.float
                ) * self.image_value_range[0]
                batch_label_gender = np.ones(
                    shape=(len(batch_files), 2),
                    dtype=np.float
                ) * self.image_value_range[0]
                for i, label in enumerate(batch_files):
                    label = int(str(batch_files[i]).split('/')[-1].split('_')[0])
                    if 0 <= label <= 5:
                        label = 0
                    elif 6 <= label <= 10:
                        label = 1
                    elif 11 <= label <= 15:
                        label = 2
                    elif 16 <= label <= 20:
                        label = 3
                    elif 21 <= label <= 30:
                        label = 4
                    elif 31 <= label <= 40:
                        label = 5
                    elif 41 <= label <= 50:
                        label = 6
                    elif 51 <= label <= 60:
                        label = 7
                    elif 61 <= label <= 70:
                        label = 8
                    else:
                        label = 9
                    batch_label_age[i, label] = self.image_value_range[-1]
                    gender = int(str(batch_files[i]).split('/')[-1].split('_')[1])
                    batch_label_gender[i, gender] = self.image_value_range[-1]

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                # update
                _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, TV = self.session.run(
                    fetches = [
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        self.EG_loss,
                        self.E_z_loss,
                        self.D_z_loss_z,
                        self.D_z_loss_prior,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input,
                        self.tv_loss
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                        self.z_prior: batch_z_prior
}

                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f" %
                    (epoch+1, num_epochs, ind_batch+1, num_batches, EG_err, TV))
                print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.age: batch_label_age,
                        self.gender: batch_label_gender,
                        self.z_prior: batch_z_prior
                    }
                )
                self.writer.add_summary(summary, self.EG_global_step.eval())

            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch+1)
            self.sample(sample_images, sample_label_age, sample_label_gender, name)
            self.test(sample_images, sample_label_gender, name)

            # save checkpoint for each 5 epoch
            if np.mod(epoch, 5) == 4:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()


    def encoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.compat.v1.get_variable_scope().reuse_variables()

        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        
        # Add this line to add the batch dimension
        current = tf.expand_dims(image, axis=0)

        # Convolutional layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = tf.keras.layers.Conv2D(
                filters=self.num_encoder_channels * (2 ** i),
                kernel_size=self.size_kernel,
                strides=2,
                padding='same',
                activation='relu',
                name=name
            )(current)


    def generator(self, z, y, gender, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.compat.v1.get_variable_scope().reuse_variables()

        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)

        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.num_categories)
        else:
            duplicate = 1

        if z is None or y is None:
            raise ValueError("Input tensors z and y cannot be None.")

        print(f"z shape: {z.shape}, y shape: {y.shape}")

        z = concat_label(z, y, duplicate=duplicate)

        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1

        if z is None or gender is None:
            raise ValueError("Input tensors z and gender cannot be None.")

        print(f"z shape after concat: {z.shape}, gender shape: {gender.shape}")

        z = concat_label(z, gender, duplicate=duplicate)

        size_mini_map = int(self.size_image / 2 ** num_layers)


        # Fully connected layer
        name = 'G_fc'
        current = tf.keras.layers.Dense(
            units=self.num_gen_channels * size_mini_map * size_mini_map,
            activation='relu',
            name=name
        )(z)

        # Reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        current = tf.nn.relu(current)

        # Deconvolutional layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = tf.keras.layers.Conv2DTranspose(
                filters=int(self.num_gen_channels / 2 ** (i + 1)),
                kernel_size=self.size_kernel,
                strides=2,
                padding='same',
                activation='relu',
                name=name
            )(current)

        name = 'G_deconv' + str(i+1)
        current = tf.keras.layers.Conv2DTranspose(
            filters=int(self.num_gen_channels / 2 ** (i + 2)),
            kernel_size=self.size_kernel,
            strides=1,
            padding='same',
            activation='relu',
            name=name
        )(current)

        name = 'G_deconv' + str(i + 2)
        current = tf.keras.layers.Conv2DTranspose(
            filters=self.num_input_channels,
            kernel_size=self.size_kernel,
            strides=1,
            padding='same',
            activation='tanh',
            name=name
        )(current)

        # Output
        return current




    def discriminator_z(self, z, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16), enable_bn=True):
        if reuse_variables:
            tf.compat.v1.get_variable_scope().reuse_variables()

        current = z

        # Fully connected layer
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_z_fc' + str(i)
            current = tf.keras.layers.Dense(
                units=num_hidden_layer_channels[i],
                name=name
            )(current)

            if enable_bn:
                name = 'D_z_bn' + str(i)
                current = tf.keras.layers.BatchNormalization(
                    scale=False,
                    training=is_training,
                    name=name
                )(current)

            current = tf.nn.relu(current)

        # Output layer
        name = 'D_z_fc' + str(i+1)
        current = tf.keras.layers.Dense(
            units=1,
            name=name
        )(current)

        return tf.nn.sigmoid(current), current


    def discriminator_img(self, image, y, gender, is_training=True, reuse_variables=False, num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if reuse_variables:
            tf.compat.v1.get_variable_scope().reuse_variables()

        num_layers = len(num_hidden_layer_channels)
        current = image

        # Convolutional layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = conv2d(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[i],
                size_kernel=self.size_kernel,
                name=name
            )

            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.keras.layers.BatchNormalization(
                    scale=False,
                    training=is_training,
                    name=name
                )(current)

            current = tf.nn.relu(current)

            if i == 0:
                current = concat_label(current, y)
                current = concat_label(current, gender, int(self.num_categories / 2))

        # Fully connected layers
        name = 'D_img_fc1'
        current = tf.keras.layers.Flatten()(current)
        current = tf.keras.layers.Dense(
            units=1024,
            name=name
        )(current)
        current = lrelu(current)

        name = 'D_img_fc2'
        current = tf.keras.layers.Dense(
            units=1,
            name=name
        )(current)

        # Output
        return tf.nn.sigmoid(current), current


    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        self.saver.save(
            self.session,
            model_path,
            global_step=self.EG_global_step.eval()
        )


    def load_checkpoint(self, model_path=None):
        if model_path is None:
            print("\n\tLoading pre-trained model ...")
            checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        else:
            print("\n\tLoading init model ...")
            checkpoint_dir = model_path

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            try:
                self.saver.restore(self.session, latest_checkpoint)
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
        else:
            print("No checkpoints found.")
            return False

    def sample(self, images, labels, gender, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: images,
                self.age: labels,
                self.gender: gender
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )


    def test(self, images, gender, name):
        test_dir = os.path.join(self.save_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        images = images[:int(np.sqrt(self.size_batch)), :, :, :]
        gender = gender[:int(np.sqrt(self.size_batch)), :]
        size_sample = images.shape[0]
        labels = np.arange(size_sample)
        labels = np.repeat(labels, size_sample)
        query_labels = np.ones(
            shape=(size_sample ** 2, size_sample),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]
        query_images = np.tile(images, [self.num_categories, 1, 1, 1])
        query_gender = np.tile(gender, [self.num_categories, 1])

        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: query_images,
                self.age: query_labels,
                self.gender: query_gender
            }
        )

        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=query_images,
            save_path=os.path.join(test_dir, 'input.png'),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_sample, size_sample]
        )


    def custom_test(self, testing_samples_dir):
        if not self.load_checkpoint():
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        file_names = glob(testing_samples_dir)
        if len(file_names) < num_samples:
            print('The number of testing images is must larger than %d' % num_samples)
            exit(0)
        sample_files = file_names[0:num_samples]
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        gender_male = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        gender_female = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(gender_male.shape[0]):
            gender_male[i, 0] = self.image_value_range[-1]
            gender_female[i, 1] = self.image_value_range[-1]

        self.test(images, gender_male, 'test_as_male.png')
        self.test(images, gender_female, 'test_as_female.png')

        print('\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png'))

