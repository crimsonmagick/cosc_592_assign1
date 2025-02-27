from abc import ABC, abstractmethod

import keras
from keras import layers, models, Input, Model
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets
import os
import time

import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_EXTENSION = 'png'

class ModelRunner(ABC):
    def __init__(self, path):
        self.path = path
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) != 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        if len(logical_gpus) != 0:
            self.device = logical_gpus[0].name
            print(f"Using GPU {self.device}")
        else:
            devices = tf.config.list_logical_devices()
            if len(devices) == 0:
                error_message = "No device found"
                print(error_message)
                raise Exception(error_message)
            self.device = devices[0].name
            print(f"Using device {self.device}")
    
    
    def _train_model(self, model, train_images, train_labels, test_images, test_labels, epoch_count, test_name, show_graph=False):
        with tf.device(self.device):
            train_images_normalized = train_images.astype('float32') / 255.0
            test_images_normalized = test_images.astype('float32') / 255.0
            start = time.time()
            history = model.fit(train_images_normalized, train_labels,
                                epochs=epoch_count,
                                batch_size=64,
                                validation_split=0.1)
            duration_ms = (time.time() - start) * 1000
            train_accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
            
            epochs = range(1, len(train_accuracy) + 1)
            test_loss, test_acc = model.evaluate(test_images_normalized, test_labels, verbose=2)
            
            os.makedirs(self.path, exist_ok=True)
            result_str = f'{test_name}: Test accuracy: {test_acc:.4f}, time: {duration_ms:.0f}ms'
            with open(f'{self.path}/results.txt', 'a+') as result_file:
                result_file.write(result_str + '\n')
            print(result_str)
            
            plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
            plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
            plt.title(test_name)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{self.path}/{test_name}.{IMAGE_EXTENSION}')
            if show_graph:
                plt.show()
            plt.clf()
    
    @abstractmethod
    def run_test(self):
        """Run the NN test"""
        
class ResCnn(ModelRunner):
    
    @staticmethod
    def build_res_block(preceding_layers):
        block = layers.Conv2D(128, (1, 1),  padding='valid')(preceding_layers)
        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)
        block = layers.Conv2D(128, (3, 3), padding='same')(block)
        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)
        block = layers.Conv2D(256, (1, 1),  padding='valid')(block)
        block = layers.BatchNormalization()(block)
        block = layers.Add()([block, preceding_layers]) # res/skip connection
        return layers.ReLU()(block)
    
    def run_test(self):
        epoch_count = 50
        test_name = "CNN With Haphazardly Placed Residual Blocks"
        
        cifar10 = datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        inputs = Input(shape=(32, 32, 3))
        outputs = layers.ZeroPadding2D(padding=(3, 3))(inputs)
        outputs = layers.Conv2D(256, (3, 3), activation='relu', padding='valid')(outputs)
        outputs = layers.MaxPooling2D((2, 2))(outputs)
        outputs = self.build_res_block(outputs)
        outputs = self.build_res_block(outputs)
        outputs = self.build_res_block(outputs)
        outputs = layers.AvgPool2D((2, 2))(outputs)
        outputs = layers.Flatten()(outputs)
        outputs = layers.Dense(512, activation='relu')(outputs)
        outputs= layers.Dense(10, activation='softmax')(outputs)
        model = Model(inputs, outputs)
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),
                      loss=SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        print(model.summary())
        self._train_model(model, train_images, train_labels, test_images, test_labels, epoch_count, test_name,
                          show_graph=True)


if __name__ == '__main__':
    ResCnn('../../standalone_results').run_test()
