from keras import layers, models
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets

from src.cifar_tests.model_runner import ModelRunner

class CifarSoftmaxOnly(ModelRunner):
    
    def run_test(self):
        epoch_count = 20
        test_name = "Softmax FF Layer Only (No CNN)"
        
        cifar10 = datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
        model = models.Sequential([
            layers.Flatten(),
            layers.Dense(10, activation='softmax'),
        ])
    
        model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        self._train_model(model, train_images, train_labels, test_images, test_labels, epoch_count, test_name)


if __name__ == '__main__':
    CifarSoftmaxOnly('../../report').run_test()