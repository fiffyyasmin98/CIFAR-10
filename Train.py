import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray, argmax
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np

#load train and test dataset
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()

#reshape data
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

#class name
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# scale pixels
# convert from integers to floats
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

#define cnn model
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu',kernel_initializer='he_uniform', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#compile model
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#fit model
history= cnn.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# evaluate and save
cnn.evaluate(X_test,y_test)
cnn.save('test8.h5')

#plot accuracy and loss for train and validation
fig = plt.figure(figsize=(14,7))
plt.subplot(3,1,1)
plt.plot(history.history['acc'], color='red', label="Train")
plt.plot(history.history['val_acc'], color='blue', label="Validation")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend( loc='lower right')

plt.subplot(3,1,2)
plt.plot(history.history['loss'], color='red', label="Train")
plt.plot(history.history['val_loss'], color='blue', label="Validation")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend( loc='upper right')

plt.tight_layout()

#save graph of accuracy and loss
plt.savefig("Model Accuracy and Loss2.jpg")
plt.show()