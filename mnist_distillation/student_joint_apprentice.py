import keras
from keras.datasets import mnist
from keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, merge
from keras.optimizers import SGD, Adam, RMSprop
from keras.constraints import max_norm
from keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.models import Sequential
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

nb_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# convert y_train and y_test to categorical binary values 
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Reshape them to batch_size, width,height,#channels
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize the values
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# Concatenate so that this becomes a 10 + 10 dimensional vector
Y_train_new = np.concatenate([Y_train, Y_train], axis=1)
Y_test_new =  np.concatenate([Y_test, Y_test], axis=1)

input_shape = (28, 28, 1) # Input shape of each image

teacher = Sequential()
teacher.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
teacher.add(Conv2D(64, (3, 3), activation='relu'))
teacher.add(MaxPooling2D(pool_size=(2, 2)))

teacher.add(Dropout(0.25)) # For reguralization

teacher.add(Flatten())
teacher.add(Dense(128, activation='relu'))
teacher.add(Dropout(0.5)) # For reguralization

teacher.add(Dense(nb_classes))
teacher.add(Activation('softmax')) # Note that we add a normal softmax layer to begin with

teacher.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print(teacher.summary())

# Student model that is stand-alone. We will evaluate its accuracy compared to a teacher trained student model
student = Sequential()
student.add(Flatten(input_shape=input_shape))
student.add(Dense(32, activation='relu'))
student.add(Dropout(0.2))
student.add(Dense(nb_classes))
student.add(Activation('softmax'))

student.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

print(student.summary())

#Training Together

student_layer = student.layers[-1].output # This is going to be a tensor. And hence it needs to pass through a Activation layer
teacher_layer =  teacher.layers[-1].output

output = concatenate([student_layer, teacher_layer])

# This is our new student model
student_teacher = Model( [student.input,teacher.input] , output )
student_teacher.summary()

# Declare knowledge distillation loss
def knowledge_distillation_loss(y_true, y_pred, alpha,beta,gamma):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    #y_true, y_logits = y_true[: , :nb_classes], y_true[: , nb_classes:]
    y_true = y_true[: , :nb_classes]
    
    y_pred_student , y_pred_teacher = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    
    #teacher , student , apprentice
    loss = ( alpha*logloss(y_true,y_pred_teacher) ) + ( beta*logloss(y_true, y_pred_student) ) + ( gamma*logloss(y_pred_teacher, y_pred_student) )
   
    return loss


# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)

def acc_teacher(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, nb_classes:]
    return categorical_accuracy(y_true, y_pred)
  
# For testing use regular output probabilities - without temperature
student_teacher.compile(
    #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
    optimizer='adadelta',
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 1,0.5,0.5),
    #loss='categorical_crossentropy',
    metrics=[acc,acc_teacher] )#,true_loss,logits_loss


epochs = 500
batch_size = 256

log_dir = 'logs/loss_joint_apprentice/'
logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-acc{acc:.4f}-acc_teacher{acc_teacher:.4f}-val_acc{val_acc:.4f}-val_acc_teacher{val_acc_teacher:.4f}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)

history = student_teacher.fit([X_train,X_train]  , Y_train_new,
                      batch_size=256,
                      epochs=epochs,
                      verbose=1,
                      validation_data=([X_test,X_test] , Y_test_new),
            callbacks=[logging,checkpoint])

last_acc = history.history['acc'][-1]
last_val_acc = history.history['val_acc'][-1]
last_loss = history.history['loss'][-1]
last_val_loss = history.history['val_loss'][-1]

hist = "acc{0:.4f}-val_acc{1:.4f}-loss{2:.4f}-val_loss{3:.4f}".format(last_acc,last_val_acc,last_loss,last_val_loss)
student_teacher.save_weights(log_dir + "last_"+ hist + ".h5")


