from keras.applications.xception import Xception
from keras.layers import  Activation,GlobalAveragePooling2D,Dense, Input, Dropout
from keras.models import Model
from keras.regularizers import l2

def build_xception(input_size=299,num_class=10,weight_decay=1e-5, dropout=0.25):
    input_tensor = Input(shape=(input_size, input_size, 3))
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        )

    x =  base_model.output
    x = Dropout(dropout)(x)
    x = GlobalAveragePooling2D()(x)
    logits = Dense(num_class, kernel_regularizer=l2(weight_decay))(x)
    probabilities = Activation('softmax')(logits)
    model = Model(inputs=input_tensor, outputs=probabilities)
    
    #for layer in model.layers[:-2]:
    #    layer.trainable = False
    
    return model