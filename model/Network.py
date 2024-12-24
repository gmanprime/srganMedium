# title           :Network.py
# description     :Architecture file(Generator)
# author          :Fang Wang
# date            :2022/2/11
# usage           :from Network import Generator
# python_version  :3.7.4

# Modules
from tensorflow import keras
from keras.layers import Activation, BatchNormalization, UpSampling2D, Flatten
from keras.layers import Dense, Input, Conv2D, LeakyReLU, PReLU, add
from keras.models import Model
from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Softmax

# from tensorflow.keras.layers import AveragePooling2D
# import tensorflow_addons as tfa


# Residual block
def res_block_gen(model, kernal_size, filters, strides, initializer):
    """
    this function is used to create a residual block for generator model

    residual block is used to learn the residual between the input and the desired output

    a residual is the difference between the input and the desired output

    Args:
        model (keras model): input model
        kernal_size (int): size of the kernel
        filters (int): number of filters
        strides (int): strides
        initializer (keras initializer): initializer
    Returns:
        keras model: model with residual
    """
    gen = model

    # this is the first layer of the residual block
    # it is a convolutional layer with the given parameters
    # looks to be a type of convolutional layer
    model = Conv2D(
        filters=filters,
        kernel_size=kernal_size,
        strides=strides,
        padding="same",
        kernel_initializer=initializer,
    )(model)

    # this is a batch normalization layer
    # it is taking in the previous covolutional layer and building a
    # batch normalization layer on top of it
    model = BatchNormalization(momentum=0.5)(model)

    # Using Parametric ReLU
    # this is a PReLU activation layer and takes in the two preceding layers as input
    # and builds on top of them. it is a type of activation layer
    model = PReLU(
        alpha_initializer="zeros",
        alpha_regularizer=None,
        alpha_constraint=None,
        shared_axes=[1, 2],
    )(model)

    # this is the second layer of the residual block
    # it is a convolutional layer with the given parameters
    # looks to be a type of convolutional layer built on top of the first series of layers
    model = Conv2D(
        filters=filters,
        kernel_size=kernal_size,
        strides=strides,
        padding="same",
        kernel_initializer=initializer,
    )(model)

    # this is a second batch normalization layer built on top of the second convolutional layer
    model = BatchNormalization(momentum=0.5)(model)

    # here we are using add to add the original input to the output of the second batch normalization layer
    # this is the residual
    # it is also the final layer of the residual block
    model = add([gen, model])

    return model


# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
class Generator(object):

    """
    Generator model for SRGAN

    Args:
        object (class): parent class
    """

    def __init__(self, noise_shape):

        self.noise_shape = noise_shape

    def generator(self):
        init = RandomNormal(stddev=0.02)

        gen_input = Input(shape=self.noise_shape)
        model = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(gen_input)
        model = PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=[1, 2],
        )(model)

        gen_model = model

        # Using 16 Residual Blocks
        for index in range(1):
            model = res_block_gen(model, 3, 64, 1, init)

        model = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(model)
        model = BatchNormalization(momentum=0.5)(model)
        model = add([gen_model, model])

        model = Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(model)

        # Task1 for classification
        model1 = Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(model)
        model1 = Conv2D(
            filters=4, kernel_size=3, strides=1, padding="same", kernel_initializer=init
        )(model1)
        output1 = Softmax()(model1)

        # Task2 for downscaling with 3 upsampling blocks
        model2 = Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(model)
        model2 = UpSampling2D(size=2)(model2)
        model2 = PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=[1, 2],
        )(model2)

        model2 = Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(model2)
        model2 = UpSampling2D(size=3)(model2)
        model2 = PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=[1, 2],
        )(model2)

        model2 = Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=init,
        )(model2)
        model2 = UpSampling2D(size=2)(model2)
        model2 = PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=[1, 2],
        )(model2)

        output2 = Conv2D(
            filters=1, kernel_size=9, strides=1, padding="same", kernel_initializer=init
        )(model2)
        #    model = Activation('tanh')(model)

        generator_model = Model(inputs=gen_input, outputs=[output1, output2])

        return generator_model


# model_gen=Generator((13, 16, 1)).generator()
# model_gen.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model_gen)
