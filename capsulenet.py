"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 100
       python CapsNet.py --epochs 100 --num_routing 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Still under-fitting.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask, k_categorical_accuracy

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y1 = layers.Input(shape=(n_class,))
    y2 = layers.Input(shape=(n_class,))
    masked_by_y1 = Mask()([digitcaps, y1])  # The true label is used to mask the output of capsule layer. For training
    masked_by_y2 = Mask()([digitcaps, y2])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    mnist_shape = (28, 28, 1)
    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(mnist_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=mnist_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y1, y2], [out_caps, decoder(masked_by_y1), decoder(masked_by_y2)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


import glob
from keras.datasets import fashion_mnist


def loader(path, batch_size=64, normalize=True):
    """Generator to be used with model.fit_generator()"""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    while True:
        files = glob.glob(os.path.join(path, '*.npz'))
        np.random.shuffle(files)
        for npz in files:
            # Load pack into memory
            archive = np.load(npz)
            x = archive['x'].reshape(-1, 42, 42, 1)
            idx = archive['idx']
            del archive
            x1 = x_train[idx[:, 0]].reshape(-1, 28, 28, 1)
            x2 = x_train[idx[:, 1]].reshape(-1, 28, 28, 1)
            y1 = y_train[idx[:, 0]]
            y2 = y_train[idx[:, 1]]
            y2_idx = y2.copy()
            y1 = to_categorical(y1, 10)
            y2 = to_categorical(y2, 10)
            # Combine y1 and y2
            y = y1.copy()
            y[np.arange(len(y)), y2_idx] = 1
            # _shuffle_in_unison(images, offsets)
            # Split into mini batches
            num_batches = int(len(x) / batch_size)
            # print(x.shape, x1.shape, x2.shape)
            x = np.array_split(x, num_batches)
            x1 = np.array_split(x1, num_batches)
            x2 = np.array_split(x2, num_batches)
            # print(y.shape, y1.shape, y2.shape)
            y = np.array_split(y, num_batches)
            y1 = np.array_split(y1, num_batches)
            y2 = np.array_split(y2, num_batches)
            while x:
                b_x = x.pop()
                b_x1 = x1.pop()
                b_x2 = x2.pop()
                b_y = y.pop()
                b_y1 = y1.pop()
                b_y2 = y2.pop()
                if normalize:
                    b_x = b_x / 255.
                    b_x1 = b_x1 / 255.
                    b_x2 = b_x2 / 255.
                yield [b_x, b_y1, b_y2], [b_y, b_x1, b_x2]


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.95 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse', 'mse'],
                  loss_weights=[1., args.lam_recon, args.lam_recon],
                  metrics={'capsnet': k_categorical_accuracy})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction,
                                           horizontal_flip=True)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    num_samples = 100000
    model.fit_generator(generator=loader('dataset', args.batch_size),
                        steps_per_epoch=num_samples//args.batch_size,
                        epochs=args.epochs,
                        callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=False)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from utils import combine_images
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon.png")
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon.png", ))
    plt.show()


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    # y_train = to_categorical(y_train.astype('float32'))
    # y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import numpy as np
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float)  # 784 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=3, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model = CapsNet(input_shape=(42,42,1),
                                n_class=10,
                                num_routing=args.num_routing)
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if args.is_training:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test))
