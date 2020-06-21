# Authors: Carlos Henrique Ponciano da Silva & Vinicius Luis da Silva
# Base article: Adam Rosembrock - https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import layers, optimizers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import functions

_labelBinarizer = LabelBinarizer()
_train_data_augmentation = ImageDataGenerator(rotation_range=20, zoom_range=0.2)

def get_model_path(filepath='\\models\\transferlearning_weights.hdf5'):
    return functions.get_current_path() + filepath

def get_callbacks(alpha, _verbose=1):
    _checkpoint = ModelCheckpoint(get_model_path(), monitor='val_acc', save_best_only=True, mode='max', verbose=_verbose,)
    _plateau = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=alpha, patience=5, verbose=_verbose)
    return [_checkpoint, _plateau]

def apply_data_augmentation(x_train, y_trian, batch_size):
    global _train_data_augmentation
    _train_data_augmentation.fit(x_train)
    return _train_data_augmentation.flow(x_train, y_trian, batch_size=batch_size)

def normalize(images, labels):
    global _labelBinarizer
    images = np.array(images) / 255.0
    labels = np.array(labels)
    labels = _labelBinarizer.fit_transform(labels)
    labels = to_categorical(labels)
    return images, labels

def load_base_network(input_shape, _set_trainable=False, _trainable_layer='block5_conv1'):
    _base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    _base.trainable = True

    for layer in _base.layers:
        if layer.name == _trainable_layer:
            _set_trainable = True
        if _set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    
    return _base

def build_model(input_shape, _percentage_droput=0.6):
    _base_network = load_base_network(input_shape)
    _model = models.Sequential()
    _model.add(_base_network)
    _model.add(layers.GlobalAveragePooling2D())
    _model.add(layers.BatchNormalization())
    _model.add(layers.Dense(128, activation='relu'))
    _model.add(layers.Dropout(0.6))
    _model.add(layers.Dense(2, activation='softmax'))
    _model.summary()
    _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return _model

def training(dataset='\\dataset\\', batch_size=32, input_shape=(150, 150, 3), random_state=42, alpha=1e-5, epochs=100):
    # try:
        print('Training neural network...')
        _images, _labels = functions.select_dataset(functions.get_current_path() + dataset)
        _images, _labels = normalize(_images, _labels)
        _callbacks = get_callbacks(alpha)
        (_x_train, _x_test, _y_train, _y_test) = train_test_split(_images, _labels, test_size=0.20, stratify=_labels, random_state=random_state)
        _data_augmentation = apply_data_augmentation(_x_train, _y_train, batch_size)
        _model = build_model(input_shape)
        _model.fit_generator(_data_augmentation,
                            steps_per_epoch=len(_x_train) // batch_size,
                            validation_data=(_x_test, _y_test),
                            validation_steps=len(_x_test) // batch_size,
                            callbacks=_callbacks,
                            epochs=epochs)
        print('Neural network training completed!')
        return True
    # except Exception as e:
    #     print(f'Training error: {e}')
    #     return False

def prediction(path):
    try: 
        _model = models.load_model(get_model_path())
        _image = np.array([functions.select_image(path)]) / 255.0
        _predict = _model.predict(_image)
        return functions.prediction_output(_predict[0])
    except Exception as e:
        print(f'prediction error: {e}')
        return None, None, e
   