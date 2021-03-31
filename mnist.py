from zipfile import ZipFile
import os
import urllib
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Mnist:
    def get_data(self, data_type):
        URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
        FILE = 'fashion_mnist_images.zip'
        FOLDER = 'data'
        CHECK_FILE = "data/"+data_type

        if not os.path.exists(CHECK_FILE):
            urllib.request.urlretrieve(URL, FILE)
            with ZipFile(FILE) as zip_images:
                zip_images.extractall(FOLDER)

        labels = os.listdir('data/'+data_type)

        X = []
        y = []
        for label in labels:
            for file in os.listdir(os.path.join('data', data_type, label)):
                image = cv2.imread(os.path.join('data/'+data_type, label, file), cv2.IMREAD_UNCHANGED)
                X.append(image)
                y.append(label)
        return np.array(X), np.array(y).astype('uint8')

    def get_train_data(self):
        return self.get_data("train")

    def get_test_data(self):
        return self.get_data("test")