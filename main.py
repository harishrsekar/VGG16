from __future__ import print_function

import numpy as np

import pandas as pd

import os
# import cv2
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def VGGupdated(input_tensor=None, classes=2):
    img_rows, img_cols = 300, 300  # by default size is 224,224
    img_channels = 3

    img_dim = (img_rows, img_cols, img_channels)

    img_input = Input(shape=img_dim)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)  # Conv1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)  # Conv2
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)  # Conv3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)  # Conv4
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)  # Conv5
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)  # Conv6
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)  # Conv7
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)  # Conv8
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)  # Conv9
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)  # Conv10
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)  # Conv11
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)  # Conv12
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)  # Conv13
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)  # Conv14
    x = Dense(4096, activation='relu', name='fc2')(x)  # Conv15
    x = Dense(classes, activation='softmax', name='predictions')(x)  # Conv16

    # Create model.

    model = Model(inputs=img_input, outputs=x, name='VGGdemo')

    return model


model = VGGupdated(classes=2)  # replace integer with number of classes you are training for
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

dataset_path = os.listdir(r'C:\Users\Harish\PycharmProjects\VGG16\venv\training_images')

data_types = os.listdir(r'C:\Users\Harish\PycharmProjects\VGG16\venv\training_images')
print(data_types)

print("Types of rooms found: ", len(dataset_path))

items = []

for item in data_types:
    all_rooms = os.listdir(r'C:\Users\Harish\PycharmProjects\VGG16\venv\training_images' + '/' + item)

# Add them to the list

for room in all_rooms:
    items.append((item, str('rooms_dataset' + '/' + item) + '/' + room))
    print(items)

# Build a dataframe
rooms_df = pd.DataFrame(data=rooms, columns=['room type', 'image'])
print(rooms_df.head())

print("Total number of rooms in the dataset: ", len(rooms_df))

room_count = rooms_df['room type'].value_counts()

print("rooms in each category: ")
print(room_count)

path = 'rooms_dataset/'

im_size = 300

images = []
labels = []

for i in data_types:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]

    for f in filenames:
        img = cv2.imread(data_path + '/' + f)
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)

images = np.array(images)

images = images.astype('float32') / 255.0
print(images.shape)

y = rooms_df['room type'].values

y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)
print(y)

y = y.reshape(-1, 1)
onehotencoder = OneHotEncoder(categories=[0])  # Converted  scalar output into vector output where the correct class will be 1 and other will be 0
Y = onehotencoder.fit_transform(y)


images, Y = shuffle(images, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, Y, test_size=0.05, random_state=415)

# inpect the shape of the training and testing.
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

model.fit(train_x, train_y, epochs=10, batch_size=32)

preds = model.evaluate(test_x, test_y)
print("Loss = " + str(preds[0]))
