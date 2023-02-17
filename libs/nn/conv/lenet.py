from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Rescaling
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class LeNet:
    @staticmethod
    def build(width:int, height:int, depth:int, classes:int):
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # initialize the model
        return Sequential([

            Rescaling(1./255,input_shape=inputShape),

            # first set of CONV => RELU => POOL layers
            Conv2D(20, (5, 5), padding='same',  activation='relu'),#,input_shape=inputShape),

            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # second set of CONV => RELU => POOL layers
            Conv2D(50, (5, 5), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            Dense(500, activation='relu'),
            Dense(classes, activation='softmax'),
        ])
