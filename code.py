# Download Dataset :

# 2017:
# !wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip
# !unzip /content/ISIC-2017_Training_Data.zip
# !wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip
# !unzip /content/ISIC-2017_Test_v2_Data.zip
# !wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv
# !wget https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv
# 
# 2016:
#  https://isic-challenge-data.s3.amazonaws.com/2016/
# 
# 
#  convert 2017--> 2016

# installation 

# !pip install mealpy
# !pip install PyWavelets
# !pip install tensorflow-wavelets
# !pip install keras-self-attention
# !pip install plot_keras_history


# import libraries

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
from glob import glob
from sklearn.metrics import roc_curve, auc
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  precision_score, recall_score, accuracy_score
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from  matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate,Dense, Conv2D, MaxPooling2D, Flatten,Input,Activation,add,AveragePooling2D,BatchNormalization,GlobalAveragePooling2D,Dropout
from sklearn.metrics import  precision_score, recall_score, accuracy_score,classification_report ,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
%matplotlib inline
import numpy as np
import math
import cv2
import pywt
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
import seaborn as sb
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout, Lambda, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Layer
import numpy as np
import keras,glob,os
import cv2
from tensorflow import keras
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DMWT as DMWT
import tensorflow_wavelets.Layers.Threshold as Threshold
import tensorflow as tf
import pywt
from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.keras import layers, initializers
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
IMG_SIZE = 150
from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqSelfAttention
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from plot_keras_history import show_history, plot_history
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Activation, Multiply, LeakyReLU, Input
from tensorflow.keras.models import Model
from keras_self_attention import SeqSelfAttention
from sklearn.preprocessing import LabelEncoder
from mealpy.utils.space import FloatVar

from tensorflow.keras import backend as K
from mealpy.optimizer import Optimizer
from mealpy import FloatVar,FOX,AGTO,GWO
from tensorflow.keras.applications import Xception

# Make label
# - 0  No
# - 1 Mel

train_df = pd.read_csv('ISIC-2017_Training_Part3_GroundTruth.csv')
test_df=pd.read_csv('ISIC-2017_Test_v2_Part3_GroundTruth.csv')
def add_jpg(x):
    x=x+".jpg"
    return x
train_df['new_image_id']=train_df['image_id'].apply(add_jpg)
test_df['new_image_id']=test_df['image_id'].apply(add_jpg)
def add_label1(x):
    if(x==0.0):
        return 'no'
    else:
        return 'mel'
train_df['id']=train_df['melanoma'].apply(add_label1)
test_df['id']=test_df['melanoma'].apply(add_label1)
train_df.head()
train_df.head()
test_df.head()

train_df.head()

# Read Training image

imagedict={}
def add_image1(x,y):
    main_path = '/kaggle/working/ISIC-2017_Training_Data'
    if x in os.listdir(main_path): 
        image=(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        image = (cv2.resize(image, (224, 224))) 
        image=np.array(image)
        image = image.astype('float32')/255
        if y in imagedict:
            imagedict[y].append(image)
        else:
            imagedict[y]=[image]
    return
train_df.apply(lambda x: add_image1(x['new_image_id'], x['melanoma']), axis=1)
train_df.head()

def add_image1(x,y):
    main_path = '/kaggle/working/ISIC-2017_Test_v2_Data'
    if x in os.listdir(main_path): 
        img_array = cv2.imread(os.path.join(main_path,x))  
        image=(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        image = (cv2.resize(image, (224, 224)))  
        image=np.array(image)
        image = image.astype('float32')/255
        if y in imagedict:
            imagedict[y].append(image)
        else:
            imagedict[y]=[image]
    return
test_df.apply(lambda x: add_image1(x['new_image_id'], x['melanoma']), axis=1)
test_df.head()

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
        )
for key in imagedict.keys():
    directory = os.path.join(f'./{key}')
    if not os.path.exists(directory):
        os.mkdir(directory)
    imagedict[key]=np.array(imagedict[key])
    imagedict[key]=imagedict[key].reshape(imagedict[key].shape[0], 224, 224, 3)
    image_gen = data_gen.flow(imagedict[key], batch_size=1, save_to_dir=f'./{key}',
                            save_prefix='image', save_format='jpg')
    print(len(imagedict[key]))
    total = 0
    for _image in image_gen:
        total += 1
        if total >= 5000:
            break

# Image generation for imbalence handeling
# Move image to balaced_data folder

if not os.path.exists('balanced_data'):
    os.mkdir('balanced_data')


# read data from balanced_data folder

datadir='/kaggle/working/balanced_data'
categories =['0.0',  '1.0']
training_data=[]
def create_training_data():
    print("OK")
    for category in categories:
        path = os.path.join(datadir,category)
        class_num = category
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(100,100))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()

test_df.head()

testing_data=[]
for index ,row in test_df.iterrows():
img_array = cv2.imread('/kaggle/working/ISIC-2017_Test_v2_Data/'+row['new_image_id'])
new_array = cv2.resize(img_array,(100,100))
testing_data.append([new_array,row['melanoma']])

Train_im,Test_im=[],[]
Train_l,Test_l=[],[]
for features,label in training_data:
    Train_im.append(features)
    Train_l.append(label)
Train_im = np.array(Train_im).reshape(len(Train_im),100,100,3) # 224
for features,label in testing_data:
    Test_im.append(features)
    Test_l.append(label)
Test_im = np.array(Test_im).reshape(len(Test_im),100,100,3)

len(Train_im),len(Test_im)

wavelet = pywt.Wavelet('haar')
class DWT_Pooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DWT_Pooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DWT_Pooling, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        input_height = tf.shape(inputs)[1]
        input_width = tf.shape(inputs)[2]
        input_channels = tf.shape(inputs)[3]
        band_low = wavelet.rec_lo
        band_high = wavelet.rec_hi
        band_length = len(band_low)
        assert band_length % 2 == 0
        band_length_half = band_length // 2
        inputs_reshaped = tf.reshape(inputs, (batch_size * input_channels, input_height, input_width, 1))
        wavelet_filter = np.outer(band_low, band_low)
        wavelet_filter = wavelet_filter[:, :, np.newaxis, np.newaxis]  
        wavelet_filter = tf.convert_to_tensor(wavelet_filter, dtype=tf.float32)
        LL = tf.nn.conv2d(inputs_reshaped, wavelet_filter, strides=[1, 2, 2, 1], padding='SAME')
        LL = tf.reshape(LL, (batch_size, input_height // 2, input_width // 2, input_channels))
        return LL
    def get_config(self):
        config = super(DWT_Pooling, self).get_config()
        return config
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]//2, input_shape[3]//2)

# *Proposed model*

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors
def build_discriminator(n_routings=3):
    img = Xception(include_top=False, weights="imagenet", input_shape=(200, 200, 3))
    
    x = tf.keras.layers.Conv2D(filters=8 * 8, kernel_size=9, strides=2, padding='same', name='primarycap_conv2')(img.output)
    x=DWT.DWT(name="haar",concat=0)(x)
    for layer in img.layers:
        layer.trainable = True
        print(layer, layer.trainable)

    x = tf.keras.layers.Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)
    x = tf.keras.layers.Lambda(squash, name='primarycap_squash')(x)
    x =SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                        kernel_regularizer=keras.regularizers.l2(1e-4),
                        bias_regularizer=keras.regularizers.l1(1e-4),
                        attention_regularizer_weight=1e-4,
                        name='Attention')(x)
    x = BatchNormalization(momentum=0.8)(x)

    
    x = Flatten()(x)
    uhat = Dense(200, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
    for i in range(n_routings):
        c = Activation('softmax', name='softmax_digitcaps'+str(i))(uhat)
        c = Dense(200)(c)
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)
    x = Dense(200,activation='relu')(s_j)
    pred = Dense(1, activation='sigmoid')(x)
    return Model(img.input, pred)
discriminator = build_discriminator()
print('DISCRIMINATOR:')
discriminator.summary()
adam = tf.keras.optimizers.Adam(0.0001)
discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])

# Label encoding

enc = LabelEncoder().fit(Train_l)
Train_l = enc.transform(Train_l)
'''enc = LabelEncoder().fit(Test_l)
Test_l = enc.transform(Test_l)'''

# Train test normalization and spliting

train_image=Train_im/255
test_image=Test_im/255
X_train,X_test,y_train , y_test =train_test_split(train_image, Train_l,
                stratify=Train_l,
                test_size=0.1)

y_train=np.array(y_train,dtype=np.float64)
y_test=np.array(y_test,dtype=np.float64)

len(y_test),y_train

# Custom Learning RateScheduler

tf.keras.backend.clear_session()

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        lr = self.model.optimizer.learning_rate
        scheduled_lr = self.schedule(epoch, lr)
        self.model.optimizer.learning_rate = scheduled_lr
        print(f"\nEpoch {epoch}: Learning rate is {float(np.array(scheduled_lr))}.")


LR_SCHEDULE = [
    (3, 0.05),
    (6, 0.01),
    (9, 0.005),
    (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
        )

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["binary_crossentropy"]
            )
        )
discriminator.compile(loss = "binary_crossentropy", optimizer ='adam',
                    metrics=['acc',tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall(),tf.keras.metrics.TruePositives(),
                                tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),
                                tf.keras.metrics.FalseNegatives()])
history=discriminator.fit(X_train, y_train,
        batch_size=50,
        epochs=50,
        verbose=1,
        validation_data=(X_test, y_test),shuffle=True)

discriminator.evaluate(X_test, y_test, verbose=0)

def build_discriminator(filters_size, kernel_size, l2_reg, l1_reg, n_routings=3):
    img_input = Input(shape=(200, 200, 3))
    base_model = Xception(include_top=False, weights="imagenet", input_tensor=img_input)

    x = tf.keras.layers.Conv2D(filters=filters_size, kernel_size=kernel_size, strides=2, padding='same', name='primarycap_conv2')(base_model.output)
    x = DWT_Pooling()(x)

    for layer in base_model.layers:
        layer.trainable = True

    x = tf.keras.layers.Reshape(target_shape=(-1, 8), name='primarycap_reshape')(x)
    x = tf.keras.layers.Lambda(squash, name='primarycap_squash')(x)
    x = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                        bias_regularizer=tf.keras.regularizers.l1(l1_reg),
                        attention_regularizer_weight=1e-4,
                        name='Attention')(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Flatten()(x)
    uhat = Dense(200, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
    for i in range(n_routings):
        c = Activation('softmax', name='softmax_digitcaps'+str(i))(uhat)
        c = Dense(200)(c)
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)
    x = Dense(200, activation='relu')(s_j)
    pred = Dense(1, activation='sigmoid')(x)
    return Model(img_input, pred)
def fitness_function(position):
    filters_size, kernel_size, lr, l2_reg, l1_reg, batch_size, epochs = position

    filters_size = int(64)
    print(filters_size)
    kernel_size = int(kernel_size)
    batch_size = int(batch_size)
    epochs = int(epochs)
    lr = float(lr)
    l2_reg = float(l2_reg)
    l1_reg = float(l1_reg)

    discriminator = build_discriminator(filters_size, kernel_size, l2_reg, l1_reg)
    adam = tf.keras.optimizers.Adam(learning_rate=lr)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam,
                        metrics=['accuracy'])

    history = discriminator.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=0,
                                validation_data=(X_test, y_test),
                                shuffle=True)
    val_acc = history.history['val_accuracy'][-1]

    return -val_acc


problem_dict = {
    "bounds": FloatVar(
        lb=[64, 3, 1e-5, 1e-5, 1e-5, 16, 5],
        ub=[256, 9, 1e-2, 1e-2, 1e-2, 64, 50],
        name="delta"
    ),
    "minmax": "min",
    "obj_func": fitness_function
}
model =GWO.IGWO(epoch=100, pop_size=50, a_min = 0.02, a_max = 2.2)
g_best = model.solve(problem_dict)
print(f"Best Hyperparameters: {g_best.solution}")
print(f"Best Validation Accuracy: {-g_best.target}")
best_filters_size = int(g_best.solution[0])
best_kernel_size = int(g_best.solution[1])
best_lr = float(g_best.solution[2])
best_l2_reg = float(g_best.solution[3])
best_l1_reg = float(g_best.solution[4])
best_batch_size = int(g_best.solution[5])
best_epochs = int(g_best.solution[6])
discriminator = build_discriminator(best_filters_size, best_kernel_size, best_l2_reg, best_l1_reg)
adam = tf.keras.optimizers.Adam(learning_rate=best_lr)
discriminator.compile(loss='binary_crossentropy', optimizer=adam,
                    metrics=['accuracy',
                                tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall(),
                                tf.keras.metrics.TruePositives(),
                                tf.keras.metrics.TrueNegatives(),
                                tf.keras.metrics.FalsePositives(),
                                tf.keras.metrics.FalseNegatives()])
history = discriminator.fit(X_train, y_train,
                            batch_size=best_batch_size,
                            epochs=best_epochs,
                            verbose=1,
                            validation_data=(X_test, y_test),
                            shuffle=True)


