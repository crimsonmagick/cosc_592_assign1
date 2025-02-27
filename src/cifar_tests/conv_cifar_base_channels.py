from enum import Enum
from keras import layers, models
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets
from src.cifar_tests.model_runner import ModelRunner

class Channels(Enum):
    Red = 0
    Green = 1
    Blue = 2

class SingleChannelCnn(ModelRunner):
    
    @staticmethod
    def filter_rgb_channel(images, channel):
        return images[:, :, :, channel:channel + 1]
    
    def run_test(self):
        test_name_template = 'Base CNN, {0} Channel Only'
        epoch_count = 20

        cifar10 = datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        for channel in Channels:
            train_images_filtered = self.filter_rgb_channel(train_images, channel.value)
            test_images_filtered = self.filter_rgb_channel(test_images, channel.value)
            
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(10, activation='softmax'),
            ])
            
            model.compile(optimizer='adam',
                          loss=SparseCategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])
            test_name = test_name_template.format(channel.name)
            self._train_model(model, train_images_filtered, train_labels, test_images_filtered, test_labels, epoch_count, test_name)

if __name__ == '__main__':
    SingleChannelCnn('../../report').run_test()