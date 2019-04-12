#Import Functions

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D,concatenate, BatchNormalization, Dropout, Flatten, Activation
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
import keras.models as model
import Classification_User as C
from PIL import ImageFile


#Hyper-Parameters

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir(C.Directory)
if not os.path.isdir("Weights"):
    os.mkdir("Weights")
if not os.path.isdir("CSV"):
    os.mkdir("CSV")
batch_size = C.Batch_Size
img_width = C.Image_Size
img_height = C.Image_Size
epochs = C.Epochs
learn_rate = C.Learning_Rate
ngpus = C.No_Of_GPUS
nclasses =  C.No_Of_Classes
layers_frozen=C.No_Of_Frozen_Layers
model_path = './'
train_data_dir = C.Training_Directory
validation_data_dir = C.Validation_Directory
Name = "InceptionResnetV2".format(int(time.time())) 



#Image-PreProcessing

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,width_shift_range=0.3,
                                   height_shift_range=0.3,rotation_range=30,shear_range=0.5,zoom_range=.7,
                                   channel_shift_range=0.3,cval=0.5,vertical_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_data_dir, target_size=(
    img_height, img_width), batch_size=batch_size, class_mode='categorical')
train_steps = train_generator.__len__()
val_steps = validation_generator.__len__()



#Base-Model
architecture=C.Architecture
if architecture==1:
    base_model = InceptionResNetV2(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="InceptionResNetV2"
elif architecture==2:
    base_model = DenseNet121(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="DenseNet121"
elif architecture==3:
    base_model = ResNet50(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="ResNet50"
elif architecture==4:
    base_model = NASNetMobile(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="NASNetMobile"
elif architecture==5:
    base_model = MobileNet(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="MobileNet"
elif architecture==6:
    base_model = InceptionV3(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    architecture_name="InceptionV3"
else:
    print ("Wrong Architecture Input")
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nclasses, activation='softmax')(x)
pmodel = Model(base_model.input, predictions)
model = multi_gpu_model(pmodel, ngpus)
for layer in model.layers[:-layers_frozen]:
    layer.trainable = False
nadam = Nadam(lr=learn_rate)
print(f'=> creating model replicas for distributed training across {ngpus} gpus <=')
model.compile(optimizer=nadam, loss='categorical_crossentropy',metrics=['accuracy'])
print('=> done building model <=')
top_weights_path = os.path.join(
                                os.path.abspath(model_path), 'Weights/top_model_weights_'+architecture_name+'_'+str(layers_frozen)+'Frozen_Layers.h5')
final_weights_path = os.path.join(
                                  os.path.abspath(model_path), 'Weights/model_weights_'+architecture_name+'_'+str(layers_frozen)+'Frozen_Layers.h5')




#Tensor-Board

tensorboard = TensorBoard(
    log_dir='./logs'.format(Name), histogram_freq=0, write_graph=True, write_images=False)
callbacks_list = [ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
                  tensorboard, EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
print('=> created callback objects <=')
print('=> initializing training loop <=')
history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=epochs,
                              validation_data=validation_generator, validation_steps=val_steps,
                              workers=8, 
                              use_multiprocessing=True, 
                              max_queue_size=500, 
                              callbacks=callbacks_list)
print('=> loading best weights <=')
model.load_weights(final_weights_path)
print('=> saving final model <=')
Final_Weights='Weights/model_'+architecture_name+"_"+str(layers_frozen)+'Frozen_Layers.h5'
pmodel.save(os.path.join(os.path.abspath(model_path), Final_Weights))




#Load-Model  #Not Required #Comment Out All Lines When Required

new_model=tf.keras.models.load_model(final_weights_path)
#new_model.summary()



#Predictions
     
predictions=[]
img_path=C.Image_Path   #Set This To The Val Directory Path
CSV_Name=C.Csv_Name    #Set CSV Name To Be Generated
filenames=validation_generator.filenames
for i in filenames:
    img = tf.keras.preprocessing.image.load_img(img_path+i, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y=new_model.predict(x)
    predict=str(y.argmax(axis=-1))
    predict=predict.replace("[","")
    predict=predict.replace("]","")
    predict=int(predict)
    predictions.append(predict)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels[4]='None'
predicted = [labels[k] for k in predictions]
results=pd.DataFrame({"Filename":filenames,"Predictions":predicted})
actual=[]
for i in results['Filename']:
    head, sep, tail = i.partition('/')
    actual.append(head)
results['Actual']=actual
results.to_csv("CSV/"+CSV_Name)
print("CSV Has Been Generated with the name :"+CSV_Name+" inside the CSV folder in the main folder")
print("Use This CSV File With the Error_Analysis.py for Error Images Segregation and Analysis")

