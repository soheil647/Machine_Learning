# baseline model with dropout and data augmentation on the cifar10 dataset
import sys
import numpy as np
from matplotlib import pyplot
import keras
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam


# load train and test dataset
def load_dataset():
    # Loading Dataset
    train_data = np.loadtxt('TinyMNIST/trainData.csv', dtype=np.float32, delimiter=',')
    train_labels = np.loadtxt('TinyMNIST/trainLabels.csv', dtype=np.int32, delimiter=',')
    test_data = np.loadtxt('TinyMNIST/testData.csv', dtype=np.float32, delimiter=',')
    test_labels = np.loadtxt('TinyMNIST/testLabels.csv', dtype=np.int32, delimiter=',')
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Feature Selection
    tr_samples_size, _ = train_data.shape

    tr_samples_size, feature_size = train_data.shape
    te_samples_size, _ = test_data.shape
    print('Train Data Samples:', tr_samples_size,
          ', Test Data Samples', te_samples_size,
          ', Feature Size(after feature-selection):', feature_size,
          ' Input Shape is : ', train_data.shape,
          'Label Size is: ', np.unique(train_labels))

    return train_data, train_labels, test_data, test_labels


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# define cnn model
def define_model(neuron_number, optimizer, activation, early_stop):
    model = Sequential()
    model.add(Dense(neuron_number, activation=activation))
    model.add(Dense(10, activation='softmax'))

    # compile model
    model.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    # pyplot.subplot(211)
    pyplot.figure(figsize=(8, 8))
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='validatation')
    pyplot.legend()
    pyplot.show()

    # plot accuracy
    # pyplot.subplot(212)
    pyplot.figure(figsize=(8, 8))
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='validatation')
    pyplot.legend()
    pyplot.show()


numberOfNeuron = [50, 100]

for neuron_size in numberOfNeuron:
    print('####################################################################################')
    print('                     Neuron Number is :          ', neuron_size)
    print('####################################################################################')

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model(neuron_number=neuron_size, optimizer=Adam(), activation='relu', early_stop=False)
    # fit model
    history = model.fit(trainX, trainY, epochs=20, validation_split=0.2, verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    # learning curves
    summarize_diagnostics(history)

    print('Test Accuracy is :  %.3f' % (acc * 100.0))
    print('Train Accuracy is :  %.3f' % (history.history['accuracy'][-1] * 100.0))

    print("Train Confusion Matrix")
    print(confusion_matrix(trainY, np.argmax(model.predict(trainX), axis=1)))

    print("Test Confusion Matrix")
    print(confusion_matrix(testY, np.argmax(model.predict(testX), axis=1)))

    print('####################################################################################')
    print()
    print()
    print()

Activations = ['relu', 'sigmoid']

for activation in Activations:
    print('####################################################################################')
    print('                     Activation Function is :          ', activation)
    print('####################################################################################')

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model(neuron_number=100, optimizer=Adam(), activation=activation, early_stop=False)
    # fit model
    history = model.fit(trainX, trainY, epochs=20, validation_split=0.2, verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    # learning curves
    summarize_diagnostics(history)

    print('Test Accuracy is :  %.3f' % (acc * 100.0))
    print('Train Accuracy is :  %.3f' % (history.history['accuracy'][-1] * 100.0))

    print("Train Confusion Matrix")
    print(confusion_matrix(trainY, np.argmax(model.predict(trainX), axis=1)))

    print("Test Confusion Matrix")
    print(confusion_matrix(testY, np.argmax(model.predict(testX), axis=1)))

    print('####################################################################################')
    print()
    print()
    print()

Optimizers = ['sgd', 'adam']

for optimizer in Optimizers:
    print('####################################################################################')
    print('                     Optimizer is :          ', optimizer)
    print('####################################################################################')

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model(neuron_number=100, optimizer=optimizer, activation='relu', early_stop=False)
    # fit model
    history = model.fit(trainX, trainY, epochs=20, validation_split=0.2, verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    # learning curves
    summarize_diagnostics(history)

    print('Test Accuracy is :  %.3f' % (acc * 100.0))
    print('Train Accuracy is :  %.3f' % (history.history['accuracy'][-1] * 100.0))

    print("Train Confusion Matrix")
    print(confusion_matrix(trainY, np.argmax(model.predict(trainX), axis=1)))

    print("Test Confusion Matrix")
    print(confusion_matrix(testY, np.argmax(model.predict(testX), axis=1)))

    print('####################################################################################')
    print()
    print()
    print()

Stopping = [True, False]

for stopping in Stopping:
    print('####################################################################################')
    print('                     Early Stoping is :          ', stopping)
    print('####################################################################################')

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model(neuron_number=100, optimizer=Adam(), activation='relu', early_stop=stopping)
    # fit model
    if (stopping):
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        history = model.fit(trainX, trainY, epochs=20, validation_split=0.2, verbose=1, callbacks=[callback])
    else:
        history = model.fit(trainX, trainY, epochs=20, validation_split=0.2, verbose=1)

    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    # learning curves
    summarize_diagnostics(history)

    print('Test Accuracy is :  %.3f' % (acc * 100.0))
    print('Train Accuracy is :  %.3f' % (history.history['accuracy'][-1] * 100.0))

    print("Train Confusion Matrix")
    print(confusion_matrix(trainY, np.argmax(model.predict(trainX), axis=1)))

    print("Test Confusion Matrix")
    print(confusion_matrix(testY, np.argmax(model.predict(testX), axis=1)))

    print('####################################################################################')
    print()
    print()
    print()

Epochs = [20, 30]

for epoch in Epochs:
    print('####################################################################################')
    print('                     Epoch is :          ', epoch)
    print('####################################################################################')

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model(neuron_number=100, optimizer=Adam(), activation='relu', early_stop=False)
    # fit model
    history = model.fit(trainX, trainY, epochs=epoch, validation_split=0.2, verbose=1)
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    # learning curves
    summarize_diagnostics(history)

    print('Test Accuracy is :  %.3f' % (acc * 100.0))
    print('Train Accuracy is :  %.3f' % (history.history['accuracy'][-1] * 100.0))

    print("Train Confusion Matrix")
    print(confusion_matrix(trainY, np.argmax(model.predict(trainX), axis=1)))

    print("Test Confusion Matrix")
    print(confusion_matrix(testY, np.argmax(model.predict(testX), axis=1)))

    print('####################################################################################')
    print()
    print()
    print()