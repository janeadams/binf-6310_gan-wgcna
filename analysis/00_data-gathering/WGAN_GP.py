#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from ops import *
from utils import *

class WGAN_GP(object):

    model_name = "WGAN_GP"     # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, inflate_to_size, gex_size, disc_internal_size):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.inflate_to_size = inflate_to_size
        self.gex_size = gex_size
        self.disc_internal_size = disc_internal_size

        if dataset_name in ['BLA', 'CPU', 'VTA', 'HIP', 'PFC', 'NAC' ]:

            self.z_dim = z_dim         # dimension of noise-vector
            self.lambd = 10       # The higher value, the more stable, but the slower convergence
            self.disc_iters = 2     # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 1e-5
            self.beta1 = 0.5 # original paper code says its 0.9, but different optimizer

            # test
            self.sample_num = 100  # number of generated images to be saved

            # load_epidermal
            self.data_X = load_epidermal(self.dataset_name, gex_size=self.gex_size)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X[0]) // self.batch_size
        else:
            raise NotImplementedError

    def discriminator(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
        # Nerwork Architecture substituted same as in (https://www.biorxiv.org/content/10.1101/262501v2.supplementary-material)

        with tf.variable_scope("discriminator", reuse=reuse):
            # WGAN-GP 참고한 코드에 있던 batch normalization은 구현하지 않았음; 그렇게 deep하지 않기 때문
            disc_dense1 = tf.layers.dense(inputs=x,
                units=self.gex_size,
                # kernel_initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05,seed=564),
                activation=None,
                name="disc_dense1")

            disc_dense1 = lrelu(disc_dense1)
            disc_dense2 = tf.layers.dense(inputs=disc_dense1,
                units=self.disc_internal_size,
                # kernel_initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05,seed=564),
                activation=None,
                name="disc_dense2")
            disc_dense2 = lrelu(disc_dense2)
            disc_output = tf.layers.dense(inputs=disc_dense2,
                # kernel_initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05,seed=564),
                units=1,
                name="disc_output")

            return disc_output

    def generator(self, z, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
        # Nerwork Architecture substituted same as in (https://www.biorxiv.org/content/10.1101/262501v2.supplementary-material)

        with tf.variable_scope("generator", reuse=reuse):

            gen_dense1 = tf.layers.dense(inputs=z,
                units=self.z_dim,
                # kernel_initializer=tf.random_uniform_initializer(minval=-0.5,maxval=0.5,seed=564),
                activation=None,
                name="gen_dense1")

            gen_dense1 = lrelu(gen_dense1)
            gen_dense2 = tf.layers.dense(inputs=gen_dense1,
                units=self.inflate_to_size,
                # kernel_initializer=tf.random_uniform_initializer(minval=-0.5,maxval=0.5,seed=564),
                activation=None,
                name="gen_dense2")
            gen_dense2 = lrelu(gen_dense2)
            gen_output = tf.layers.dense(inputs=gen_dense2,
                units=self.gex_size,
                # kernel_initializer=tf.random_uniform_initializer(minval=-0.5,maxval=0.5,seed=564),
                activation=None,
                name="gen_output")
            gen_output = lrelu(gen_output)
            return gen_output


    def build_model(self):
        # some parameters

        bs = self.batch_size

        """ Graph Input """
        # 두 코드를 합치다 보니 일관성 없는 스타일로 설정하게 되었음, 그냥 그대로 유지함 [bs] or NaN
        self.inputs = tf.placeholder(tf.float32, [bs] + [self.gex_size] , name='real_gx')
        # noises
        self.z = tf.placeholder(tf.float32, shape=(None, self.z_dim))

        """ Loss Function """

        # output of D for real gene expression
        # 별도의 activation fn을 정의하지 않았으므로 D_real 값을 직접 loss에 넣음; 상식적으로는 logistic actication 하는게 맞는 거 같음

        D_real = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake gene expression
        G = self.generator(self.z, is_training=True, reuse=False)
        D_fake = self.discriminator(G, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = - tf.reduce_mean(D_real)
        d_loss_fake = tf.reduce_mean(D_fake)

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = - d_loss_fake

        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        alpha = tf.random_uniform(shape=self.inputs.get_shape(), minval=0.,maxval=1.)
        differences = G - self.inputs # This is different from MAGAN
        interpolates = self.inputs + (alpha * differences)
        D_inter = self.discriminator(interpolates, is_training=True, reuse=True)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.lambd * gradient_penalty

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'disc_' in var.name]
        g_vars = [var for var in t_vars if 'gen_' in var.name]

        # optimizers
        ### Thanks to taki0112 for the TF StableGAN implementation https://github.com/taki0112/StableGAN-Tensorflow
        # 9_GAN_supercom 에서 RMSprop으로 대체됨
        # from Adam_prediction import Adam_Prediction_Optimizer
        # self.g_optim = Adam_Prediction_Optimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, prediction=True).minimize(self.g_loss, var_list=g_vars)
        # self.d_optim = Adam_Prediction_Optimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, prediction=False).minimize(self.d_loss, var_list=d_vars)

        '''
        original code seems to be deceive people who use it it switched opt_d and opt_g !!!
        ### Thanks to taki0112 for the TF StableGAN implementation https://github.com/taki0112/StableGAN-Tensorflow
        from Adam_prediction import Adam_Prediction_Optimizer
        opt_g = Adam_Prediction_Optimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, prediction=True).minimize(obj_d, var_list=d_params)
        opt_d = Adam_Prediction_Optimizer(learning_rate=learn_rate, beta1=0.9, beta2=0.999, prediction=False).minimize(obj_g, var_list=g_params)
        '''


        # 9_GAN_supercom에서 복원됨
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.RMSPropOptimizer(self.learning_rate,
                                                    decay=0.9,
                                                    momentum=0.9,
                                                    epsilon=1e-10,
                                                    use_locking=False,
                                                    centered=False,
                                                    name='RMSProp') \
                      .minimize(self.d_loss, var_list=d_vars)

            self.g_optim = tf.train.RMSPropOptimizer(self.learning_rate,
                                                    decay=0.9,
                                                    momentum=0.9,
                                                    epsilon=1e-10,
                                                    use_locking=False,
                                                    centered=False,
                                                    name='RMSProp') \
                      .minimize(self.g_loss, var_list=g_vars)


        """ Testing """
        # for test
        self.fake_gex = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):

        # initialize all variables
        # Variable range setting은 하지 않음 일단 학습이 잘 안되면 설정하자
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        # 중간 결과물 저장하는 코드; 하단에서 주석처리됨
        #data_max_value = np.amax(self.data_X[0])
        #temp_norm = np.random.normal(0.0, data_max_value/10, size=(self.sample_size, self.z_dim))
        #temp_poisson = np.random.poisson(1, size=(self.sample_size, self.z_dim))
        #self.sample_z = np.abs(temp_norm + temp_poisson)

        # self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=25)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter # It's epoch ! (int)(checkpoint_counter / self.num_batches)
            start_batch_id = 0 # checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
        g_loss = 0
        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):

            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_gex = self.data_X[0][idx*self.batch_size:(idx+1)*self.batch_size]
                data_max_value = np.amax(self.data_X[0])
                temp_norm = np.random.normal(0.0, data_max_value/10, size=(self.batch_size, self.z_dim))
                temp_poisson = np.random.poisson(1, size=(self.batch_size, self.z_dim))
                batch_z = np.abs(temp_norm + temp_poisson)

                # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # update D network
                d_loss, _, summary_str = self.sess.run([self.d_loss, self.d_optim, self.d_sum],
                                               feed_dict={self.inputs: batch_gex, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                if (counter-1) % self.disc_iters == 0:
                    data_max_value = np.amax(self.data_X[0])
                    temp_norm = np.random.normal(0.0, data_max_value/10, size=(self.batch_size, self.z_dim))
                    temp_poisson = np.random.poisson(1, size=(self.batch_size, self.z_dim))
                    batch_z = np.abs(temp_norm + temp_poisson)

                    # batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                    g_loss, _, summary_str = self.sess.run([self.g_loss, self.g_optim, self.g_sum], feed_dict={self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 3000 counter
                # 중가나 결과물은 저장하지 않는 것으로 함
                # if np.mod(counter, 300) == 0:
                #     samples = self.sess.run(self.fake_gex,
                #                             feed_dict={self.z: self.sample_z})
                #     # tot_num_samples = min(self.sample_num, self.batch_size)
                #     # manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                #     # manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                #
                #     np.savetxt( check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.csv'.format(epoch, idx),
                #                 samples, delimiter=",")

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
            # 체크포인트는 100000 에폭 마다 저장, 결과물은 1000에폭 마다 저장함
            if np.mod(epoch, 1000) == 0:
                # save model
                if (np.mod(epoch, 1000) == 0) and (epoch > 39000):
                    self.save(self.checkpoint_dir, epoch)
                # show temporal results
                self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        if np.mod(epoch, 1000) == 0:

            """ random condition, random noise """

            # 500개의 샘플 생성하도록 함
            data_max_value = np.amax(self.data_X[0])
            temp_norm = np.random.normal(0.0, data_max_value/10, size=(self.sample_num, self.z_dim))
            temp_poisson = np.random.poisson(1, size=(self.sample_num, self.z_dim))
            z_sample = np.abs(temp_norm + temp_poisson)

            # z_sample = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

            samples = self.sess.run(self.fake_gex, feed_dict={self.z: z_sample})

            np.savetxt( check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_500_gex.csv',
                        samples, delimiter=",")

            # save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
            #             check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
