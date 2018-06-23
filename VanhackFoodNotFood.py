'''
Created on 23 de jun de 2018

@author: Antonyus Ferreira
'''
# import keras
import cv2
import numpy as np
from os import listdir

import keras
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras import metrics
from keras.layers import  activations

from keras.applications.inception_v3 import InceptionV3

from keras.utils.generic_utils import CustomObjectScope




training = listdir("Food-5K/training")
validation = listdir("Food-5K/validation")
test = listdir("Food-5K/evaluation")

train_data_dir = 'Food-5K/training'
validation_data_dir = 'Food-5K/validation'
test_data_dir = 'Food-5K/evaluation'

img_width = 224
img_height = 224

nb_train_samples = 3000
nb_validation_samples = 1000
nb_test_samples = 1000
epochs = 10
batch_size = 16

num_classes = 2


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# base_model = keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3),  
#                                                include_top=False, weights='imagenet', 
#                                                classes=2)

# with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
#     base_model = keras.models.load_model('weights.hdf5')

# base_model = keras.models.load_model('mobilenet_v2.h5', custom_objects={
#                    'relu6': keras.applications.mobilenet.relu6})

# base_model = InceptionV3(weights='imagenet', include_top=False)
# 
# 
# x = base_model.output
# predictions = Dense(2, activation='softmax', kernel_initializer = 'he_normal')(x)
# 
# model = Model(inputs=base_model.input, outputs=predictions)

input_shape = (img_width, img_height, 3)
   
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
   
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
   
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu', kernel_initializer = 'he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
   
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer = 'he_normal'))
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu', kernel_initializer = 'he_normal'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax', kernel_initializer = 'he_normal'))

model.summary()

# model.compile(loss=keras.losses.cosine_proximity,
#              optimizer=keras.optimizers.RMSprop(lr=learn_rate,  epsilon=1e-08, decay=learn_rate/epochs),
#               metrics=['accuracy', metrics.top_k_categorical_accuracy])

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    verbose =2,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)



model.save_weights('first_try.h5')


model.predict_generator(test_generator)

# training_dataset = image.load_img('Food-5K/training/0_0.jpg', target_size=(224, 224))
# training_dataset = image.img_to_array(training_dataset)
# training_dataset = np.expand_dims(training_dataset, axis=0)
# training_dataset = preprocess_input(training_dataset)
# 
# for file_name in training:
#     
#     img = image.load_img('Food-5K/training/'+file_name, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     
# #     try:
#     training_dataset = [training_dataset, x]
# #     except ValueError:  #raised if `y` is empty.
# #         pass

    
    
#     print("shape", training_dataset.shape)
print("''")

    





# cv2.imshow("original",img)
# 
# cv2.imshow("resided",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

