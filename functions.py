# Authors: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva
from PIL import Image
from os import listdir, path
from os.path import isdir
import numpy as np

def get_current_path():
    return path.dirname(__file__)

def select_image(filename):
    _image = Image.open(filename)
    _image = _image.convert('RGB')
    _image = _image.resize((150,150))
    return np.asarray(_image)

def load_class(directory, label_class, images, labels):
    for filename in listdir(directory):
        path = directory + filename
        try:
            images.append(select_image(path))
            labels.append(label_class)
        except:
            print(f'Error loading image {path}')
    return images, labels

def select_dataset(directory):
    images = labels = list()
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        if not isdir(path):
            continue
        images, labels = load_class(path, subdir, images, labels)
    return images, labels

def prediction_output(predict):
    print(predict)
    if predict[0] > predict[1]:
        _is_covid = True
        _percentage = predict[0]
    else:
        _is_covid = False
        _percentage = predict[1]

    return _is_covid, _percentage * 100, None