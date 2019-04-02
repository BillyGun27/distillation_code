
import keras
from keras import optimizers
from keras.layers import Activation, Input, Dense, Lambda, concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input

from utils.distillation import logits_generator , knowledge_distillation_loss , acc

from model.xception import build_xception as build_teacher
from model.mobilenet import build_mobilenet as build_student

data_dir = 'img/'
TARGET_SIZE = 224
log_dir = 'logs/student/'

teacher = build_teacher(input_size=224,num_class=10,weight_decay=1e-5, dropout=0.25)
teacher.load_weights("something.h5")

teacher = Model(teacher.input, teacher.layers[-2].output)
teacher._make_predict_function()

#Prepare Dataset 
data_generator = ImageDataGenerator(
    data_format='channels_last',
    rescale=(1/255)
    #preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    data_dir + 'train',
    target_size=(TARGET_SIZE,TARGET_SIZE),
    batch_size=64
)

val_generator = data_generator.flow_from_directory(
     data_dir + 'val',
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=64 , shuffle=False
)

test_generator = data_generator.flow_from_directory(
    data_dir + 'test',
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=1 , shuffle=False
)

#Prepare Model

student = build_student(input_size=224,num_class=10,weight_decay=1e-5, dropout=0.25)

# Now collect the logits from the last layer
logits = student.layers[-2].output # This is going to be a tensor. And hence it needs to pass through a Activation layer
probs = Activation('softmax')(logits)

# softed probabilities at raised temperature
logits_T = Lambda(lambda x: x / temp)(logits)
probs_T = Activation('softmax')(logits_T)

output = concatenate([probs, probs_T])

# This is our new student model
student = Model(student.input, output)

student.summary()

alpha = 0.1
temp = 1

student.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, alpha),
    metrics=[acc]
)

#Training

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, epsilon=0.007)
early_stopping =  EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

epochs = 500
student.fit_generator(
    logits_generator( train_generator , teacher ,temp) , 
    steps_per_epoch=STEP_SIZE_TRAIN, epochs=30, verbose=1,
    callbacks=[
        logging,checkpoint,reduce_lr
    ],
    validation_data=logits_generator( val_generator , teacher ,temp) , validation_steps=STEP_SIZE_VALID, workers=4
)
#,early_stopping

last_acc = history.history['acc'][-1]
last_val_acc = history.history['val_acc'][-1]
last_loss = history.history['loss'][-1]
last_val_loss = history.history['val_loss'][-1]

hist = "acc{0:.4f}-val_acc{1:.4f}-loss{2:.4f}-val_loss{3:.4f}".format(last_acc,last_val_acc,last_loss,last_val_loss)

model.save_weights(log_dir + "last_"+ hist + ".h5")