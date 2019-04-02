import keras
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling2D, Dropout, Dense, Input


def build_mobilenet(input_size=224, num_class=10 , weight_decay=1e-5, dropout=0.25):
    input_tensor = Input(shape=(input_size, input_size, 3))
    base_model = MobileNetV2(
        include_top=False, weights='imagenet', 
        input_tensor=input_tensor,
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    logits = Dense(num_class, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    probabilities = Activation('softmax')(logits)
    model = Model(base_model.input, probabilities)
    
    #for layer in model.layers[:-2]:
    #    layer.trainable = False
    
    return model