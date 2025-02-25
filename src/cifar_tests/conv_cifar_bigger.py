from keras import layers, models, Input
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets
from src.cifar_tests.model_runner import ModelRunner


class BiggerCnn(ModelRunner):
    
    def run_test(self):
        epoch_count = 20
        test_name = "Bigger CNN 2"
        
        cifar10 = datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
        model = models.Sequential([
            Input(shape=(32, 32, 3)),
            layers.ZeroPadding2D(
                padding=(3, 3),
            ),
            layers.Conv2D(64, (3, 3), activation='relu', padding='valid'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ])
        
        model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        print(model.summary())
        self._run_model(model, train_images, train_labels, test_images, test_labels, epoch_count, test_name,
                        show_graph=True)


if __name__ == '__main__':
    BiggerCnn('../../report').run_test()
