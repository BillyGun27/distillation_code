from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input

# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

def logits_generator(data_generator, teacher_logits , temp):
    # get x_train , y_train and return with added logits

    X_train, Y_train = next(valid_generator)

    Y_logits = teacher_logits.predict(image_data)
    Y_logits_T = m_logits / temp

    Y_logits_soft = softmax( Y_logits_T )

    Y_data_new = np.concatenate([Y_train, Y_logits_soft], axis=1)

    return X_train, Y_data_new

# Declare knowledge distillation loss
def knowledge_distillation_loss(y_true, y_pred, alpha):
    nb_classes = 10
    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_logits = y_true[: , :nb_classes], y_true[: , nb_classes:]
    
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    
    #teacher , student
    loss = ( alpha*temp*logloss(y_logits, y_pred_softs) ) + ( (1-alpha)*logloss(y_true,y_pred) ) 
    
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    nb_classes = 10
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)
