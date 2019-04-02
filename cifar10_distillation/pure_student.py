import keras
from keras.datasets import cifar10
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from model.mobilenet import build_mobilenet as build_model

nb_classes = 10

# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

nb_classes = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# convert y_train and y_test to categorical binary values 
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the values
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Student model that is stand-alone. We will evaluate its accuracy compared to a teacher trained student model
student = build_model(input_size=32,num_class=10,weight_decay=1e-5, dropout=0.25)


#sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
student.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print( student.summary() )

epochs = 500
batch_size = 256

log_dir = 'logs/pure_student/'
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-acc{acc:.4f}-val_acc{val_acc:.4f}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)

history = student.fit(X_train, Y_train,
          batch_size=256,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, Y_test),
           callbacks=[logging,checkpoint] )

last_acc = history.history['acc'][-1]
last_val_acc = history.history['val_acc'][-1]
last_loss = history.history['loss'][-1]
last_val_loss = history.history['val_loss'][-1]

hist = "acc{0:.4f}-val_acc{1:.4f}-loss{2:.4f}-val_loss{3:.4f}".format(last_acc,last_val_acc,last_loss,last_val_loss)
student.save_weights(log_dir + "last_"+ hist + ".h5")


