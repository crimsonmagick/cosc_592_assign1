from keras import layers, models
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets

from src.cifar_tests.model_runner import ModelRunner

class SoftmaxFusedCifar(ModelRunner):
    
    def run_test(self):
        epoch_count = 20
        test_name = "All Activation Functions Disabled Except Softmax"
        
        cifar10 = datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation=None, input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation=None),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation=None),
            layers.Flatten(),
            layers.Dense(64, activation=None),
            layers.Dense(10, activation='softmax'),
        ])
        
        model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        self._run_model(model, train_images, train_labels, test_images, test_labels, epoch_count, test_name)

if __name__ == '__main__':
    SoftmaxFusedCifar('../../report').run_test()