import cv2
import numpy as np
from keras import Model
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


class FeatureExtractor:

    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=True)

    def spatial_features(self, image_path):
        img_array = image.img_to_array(cv2.resize(cv2.imread(image_path), (224, 224)))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Get pre-last layer
        model_extract_features = Model(inputs=self.model.inputs, outputs=self.model.get_layer('fc2').output)

        # Extract features
        fc2_features = model_extract_features.predict(img_array)

        # Reshape the output
        fc2_features = [f[0] for f in fc2_features.reshape((4096, 1))]

        return fc2_features
