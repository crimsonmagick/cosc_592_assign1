from keras import layers, models, Input, Model
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets
from src.cifar_tests.model_runner import ModelRunner


class BiggerCnn(ModelRunner):
    
    @staticmethod
    def build_res_block(preceding_layers):
        block = layers.Conv2D(64, (1, 1),  padding='valid')(preceding_layers)
        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)
        block = layers.Conv2D(64, (3, 3), padding='same')(block)
        block = layers.BatchNormalization()(block)
        block = layers.ReLU()(block)
        block = layers.Conv2D(128, (1, 1),  padding='valid')(block)
        block = layers.BatchNormalization()(block)
        block = layers.Add()([block, preceding_layers]) # res/skip connection
        return layers.ReLU()(block)
    
    def run_test(self):
        epoch_count = 20
        test_name = "Bigger CNN 2"
        
        cifar10 = datasets.cifar10
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        inputs = Input(shape=(32, 32, 3))
        outputs = layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(inputs)
        outputs = layers.MaxPooling2D((2, 2))(outputs)
        outputs = self.build_res_block(outputs)
        outputs = self.build_res_block(outputs)
        outputs = self.build_res_block(outputs)
        outputs = layers.Flatten()(outputs)
        outputs = layers.Dense(64, activation='relu')(outputs)
        outputs= layers.Dense(10, activation='softmax')(outputs)
        model = Model(inputs, outputs)
        
        model.compile(optimizer='adam',
                      loss=SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        print(model.summary())
        self._run_model(model, train_images, train_labels, test_images, test_labels, epoch_count, test_name,
                        show_graph=True)


if __name__ == '__main__':
    BiggerCnn('../../report').run_test()
