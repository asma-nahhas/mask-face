import numpy as np
import os
import cv2
import tensorflow as tf


def img_to_encoding(image_path, model):
    print('in utility')
    img1 = cv2.imread(image_path, -1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    print('asma')
    print(img)
    x_train = np.array([img])
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        #embedding = model.predict_on_batch(x_train)
        embedding = model.predict(x_train)
    return embedding

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (96, 96))
    cv2.imwrite(image_path, img)
