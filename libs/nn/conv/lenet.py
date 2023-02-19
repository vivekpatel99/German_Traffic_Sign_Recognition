from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Rescaling, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K


class LeNet:
    @staticmethod
    def build(width:int, height:int, depth:int, classes:int):
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # initialize the model
        return Sequential([

            # first set of CONV => RELU => POOL layers
            Conv2D(20, (5, 5), padding='same',  activation='relu',input_shape=inputShape),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # second set of CONV => RELU => POOL layers
            Conv2D(50, (5, 5), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            Dense(500, activation='relu'),
            Dense(classes, activation='softmax'),
        ])
    #     return  keras.models.Sequential([    
    #     keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=inputShape),
    #     keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(2, 2)),
    #     keras.layers.BatchNormalization(axis=-1),
        
    #     keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    #     keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(2, 2)),
    #     keras.layers.BatchNormalization(axis=-1),
        
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(512, activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dropout(rate=0.5),
        
    #     keras.layers.Dense(43, activation='softmax')
    # ])
