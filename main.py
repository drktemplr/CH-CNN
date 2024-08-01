import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
from keras_cv.layers import SqueezeAndExcite2D
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import StratifiedKFold

# load dataset from tensorflow and split into 80% of training set and 20% of testing set
# configure batch_size to -1 to get full numpy arrays instead of default tuples

train = tfds.load(name='colorectal_histology', split='train[:80%]', batch_size=-1)
test = tfds.load(name='colorectal_histology', split='train[80%:]', batch_size=-1)

# convert to iterable numpy arrays to further split the image and label

train = tfds.as_numpy(train)
test = tfds.as_numpy(test)

x_train, y_train = train['image'], train['label']
x_test, y_test = test['image'], test['label']
x_train, x_test = x_train / 255.0, x_test / 255.0

# set random seed

seed = 7
np.random.seed(seed)

# define input data

X = x_train
Y = y_train

# define 10-fold cross validation

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

# build CNN model for training

def build_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),
        SqueezeAndExcite2D(64),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),
        SqueezeAndExcite2D(128),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D((2, 2)),
        SqueezeAndExcite2D(256),
        layers.Flatten(),
        layers.Dropout(0.8),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(8, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001, epsilon=0.1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model

# get model architecture information

sample_model = build_model()
sample_model.summary()

# train using k-fold crossvalidation

for train, test in kfold.split(X, Y):
    
    model = build_model()

    history = model.fit(X[train], Y[train], epochs=50, validation_data=(X[test], Y[test]), batch_size=12)

    # Get final validation accuracy in this fold

    score = history.history['val_accuracy'][-1] * 100
    cvscores.append(score)
    print("validation accuracy: %.2f%%" % score)
        
    # plot losses and accuracies in two sub-plots in a single graph

    figure, axes = plt.subplots(1, 2)
    figure.tight_layout()

    plt.suptitle("Result/epochs")
    plt.style.use('bmh')

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy', color='royalblue')
    plt.plot(history.history['val_accuracy'], label='validation accuracy', color='orange')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss', color='navy')
    plt.plot(history.history['val_loss'], label='validation loss', color='maroon')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    plt.savefig('result.png')

    plt.show()

# print avergae accuracy

print("%.2f (± .%.2f%%)" % (np.mean(cvscores), np.std(cvscores)))