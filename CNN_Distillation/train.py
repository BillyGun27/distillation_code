import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras.applications.imagenet_utils import preprocess_input

from model.xception import build_xception as build_model
data_dir = 'img/'
TARGET_SIZE = 224
log_dir = 'logs/teacher/'

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
    data_dir + 'val',
    target_size=(TARGET_SIZE, TARGET_SIZE),
    batch_size=1 , shuffle=False
)

#Prepare Model

model = build_model(input_size=224,num_class=10,weight_decay=1e-5, dropout=0.25)

model.compile(
    optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
    loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
)

#Training

logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-acc{acc:.4f}-val_acc{val_acc:.4f}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, epsilon=0.007)
early_stopping =  EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

model.fit_generator(
    train_generator, 
    steps_per_epoch=80, epochs=30, verbose=1,
    callbacks=[
        logging,checkpoint,reduce_lr,early_stopping
    ],
    validation_data=val_generator, validation_steps=80, workers=4
)
