import tensorflow as tf
import time
from keras import layers, models
from keras.src.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from tf_keras import datasets

RED = 0
GREEN = 1
BLUE = 2

def filter_rgb_channel(images, channel):
    return images[:, :, :, channel:channel + 1]

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
    
    rgb_channel = BLUE

    train_images = filter_rgb_channel(train_images, rgb_channel)
    test_images = filter_rgb_channel(test_images, rgb_channel)
    
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
    start = time.time()
    history = model.fit(train_images, train_labels,
                        epochs=50,
                        batch_size=64,
                        validation_split=0.1)
    duration_ms = (time.time() - start) * 1000
    train_accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
    
    epochs = range(1, len(train_accuracy) + 1)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc:.4f}, time: {duration_ms:.0f}ms')
    
    plt.plot(epochs, train_accuracy, 'g', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
