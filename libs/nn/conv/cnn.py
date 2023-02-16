from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K


class CNN:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # initialize the model
        return Sequential([
            # first set of CONV => RELU => POOL layers
            Conv2D(20, (5, 5), padding='same', input_shape=inputShape, activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            # second set of CONV => RELU => POOL layers
            Conv2D(50, (5, 5), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            Dense(500, activation='relu'),
            Dense(classes, activation='softmax'),
        ])
