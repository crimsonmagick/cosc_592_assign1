import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from keras.src.losses import SparseCategoricalCrossentropy
from tf_keras import datasets

gpus = tf.config.list_logical_devices('GPU')
if len(gpus) != 0:
    device = gpus[0].name
    print(f"Using GPU {device}")
else:
    devices = tf.config.list_logical_devices()
    if len(devices) == 0:
        error_message = "No device found"
        print(error_message)
        raise Exception(error_message)
    device = devices[0].name
    print(f"Using device {device}")

with tf.device(device):
    cifar10 = datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=None, input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=None),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=None),
        layers.Flatten(),
        layers.Dense(64, activation=None),
        layers.Dense(10, activation=None),
    ])
    
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels,
                        epochs=50,
                        batch_size=64,
                        validation_split=0.1)
    train_accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
    
    epochs = range(1, len(train_accuracy) + 1)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

