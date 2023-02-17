import numpy as np
import cv2
import os
import pandas  as df

class SimpleDatasetLoader:
    def __init__(self, preprocessors: list = None) -> None:
        """store the image preprocessor
        """
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, class_id:df.Series, paths:df.Series, verbose=-1) -> tuple:
        """ intialize the list of features and labels
        """
        data = []
        labels = []

        # loop over the input images
        for i, (label, imagePath) in enumerate(zip(class_id, paths)):

            image = cv2.imread(imagePath)

            # check to see if preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessor and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # theat our processed image as a 'feature vector'
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every 'verbose' images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1}/{len(paths)}")

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

