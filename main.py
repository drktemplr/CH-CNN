import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold

# load dataset from tensorflow and split into 70% of training set and 30% of testing set
# configure batch_size to -1 to get full numpy arrays instead of default tuples

train = tfds.load(name='colorectal_histology', split='train[:70%]', batch_size=-1)
test = tfds.load(name='colorectal_histology', split='train[70%:]', batch_size=-1)

# convert to iterable numpy arrays to further split the image and label

train = tfds.as_numpy(train)
test = tfds.as_numpy(test)

x_train, y_train = train['image'], train['label']
x_test, y_test = test['image'], test['label']
x_train, x_test = x_train / 255.0, x_test / 255.0

# set random seed

seed = 7
np.random.seed(seed)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# build CNN model for training

model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(8, activation='softmax')
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)

# plot losses along with accuracies in a single graph

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.legend(loc='lower left')
plt.savefig("result.png")
plt.show()