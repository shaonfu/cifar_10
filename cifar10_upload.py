#50轮迭代之后 85.8 acc
#200轮  Test loss: 0.4044540786415339
#       Test accuracy: 0.8984
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D
from lsuv_init import LSUVinit
import os

batch_size = 32
num_classes = 10
data_augmentation = True
epochs = 10
save_dir = './models'
model_name = 'keras_cifar10_trained_upload.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

model = Sequential([
    Conv2D(32,(3,3),padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(48, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(48, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Conv2D(80, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(80, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(80, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(80, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(80, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(128, (3, 3), padding='same',
           input_shape=x_train.shape[1:]),
    Activation('relu'),
    GlobalMaxPooling2D(),
    Dropout(0.25),

    Dense(500),
    Activation('relu'),
    Dropout(0.25),
    Dense(num_classes),
    Activation('softmax')
])

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#model = LSUVinit(model,x_train[:batch_size,:,:,:])
model.summary()

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None,  # image data format, either "channels_first" or "channels_last"
        validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    result = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir,model_name)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])