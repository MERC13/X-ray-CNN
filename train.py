import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Activation
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import Callback, EarlyStopping
from keras.utils import to_categorical, load_img,img_to_array
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = Path("chest_xray/chest_xray/")
train_dir = data_dir / 'train'
val_dir = data_dir / 'val'
test_dir = data_dir / 'test'

def load_train():
    normal_cases_dir = train_dir / 'NORMAL'
    abnormal_cases_dir = train_dir / 'PNEUMONIA'
    normal_cases = normal_cases_dir.glob('*.jpeg')
    abnormal_cases = abnormal_cases_dir.glob('*.jpeg')
    train_data = []
    train_label = []
    for img in normal_cases:
        train_data.append(img)
        train_label.append('NORMAL')
    for img in abnormal_cases:
        train_data.append(img)
        train_label.append('PNEUMONIA')
    df = pd.DataFrame(train_data)
    df.columns = ['images']
    df['labels'] = train_label
    df=df.sample(frac=1).reset_index(drop=True)
    return df

train_data = load_train()
# plt.bar(train_data['labels'].value_counts().index, train_data['labels'].value_counts().values)
# plt.show()

def plot(image_batch, label_batch):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        ax = plt.subplot(2, 5, i + 1)
        img = cv2.imread(str(image_batch[i]))
        img = cv2.resize(img, (224, 224))
        plt.imshow(img)
        plt.title(label_batch[i])
        plt.axis('off')
        plt.show()

# image_batch = train_data['images'].values[:10]
# label_batch = train_data['labels'].values[:10]
# plot(image_batch, label_batch)

def prepare_and_load(isval=True):
    if isval==True:
        normal_dir = val_dir / 'NORMAL'
        pneumonia_dir = val_dir / 'PNEUMONIA'
    else:
        normal_dir = train_dir / 'NORMAL'
        pneumonia_dir = train_dir / 'PNEUMONIA'
    normal_cases = normal_dir.glob('*.jpeg')
    pneumonia_cases = pneumonia_dir.glob('*.jpeg')
    data,labels=([] for x in range(2))
    # data, labels = [], []
    def prepare(case):
        for image in case:
            img = cv2.imread(str(image))
            img = cv2.resize(img, (224, 224))
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
            if case==normal_cases:
                label=to_categorical(0, num_classes=2)
            else:
                label = to_categorical(1, num_classes=2)
            data.append(img)
            labels.append(label)
        return data,labels
    prepare(normal_cases)
    d,l=prepare(pneumonia_cases)
    d=np.array(d)
    l=np.array(l)
    return d,l

val_data, val_labels = prepare_and_load(isval=True)
test_data, test_labels = prepare_and_load(isval=False)
# print('Number of test images -->', len(test_data))
# print('Number of val images -->', len(val_data))

def data_gen(data, batch_size):
    n = len(data)
    steps = n//batch_size
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype=np.float32)
    indices = np.arange(n)
    
    i=0
    while True:
        np.random.shuffle(indices)
        count=0
        next_batch=indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['images']
            label = data.iloc[idx]['labels']
            if label == 'NORMAL':
                label = 0
            else:
                label = 1
            encoded_label = to_categorical(label, num_classes=2)
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_img = img.astype(np.float32)/255.
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            count += 1
            if count==batch_size-1:
                break
        i+=1
        yield batch_data, batch_labels
        if i>=steps:
            i=0
            
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(224,224,3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())

# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('softmax'))

batch_size = 16
nb_epochs = 1

train_data_gen = data_gen(data=train_data, batch_size=batch_size)
nb_train_steps = train_data.shape[0] // batch_size

# print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(val_data)))

# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# history = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps, validation_data=(val_data, val_labels))

def vgg16_model(num_classes=None):
    model=VGG16(weights='imagenet',include_top=True, input_shape=(224,224,3))
    
    x=Dense(1024,activation='relu')(model.layers[-4].output)
    x=Dropout(0.7)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(2,activation='softmax')(x)
    model=Model(model.inputs,x)
    return model

vgg_conv=vgg16_model(2)
for layer in vgg_conv.layers[:-10]:
    layer.trainable=False
    
opt = Adam(lr=0.0001, decay=1e-5)
vgg_conv.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

history = vgg_conv.fit_generator(train_data_gen, epochs=1, steps_per_epoch=nb_train_steps, validation_data=(val_data, val_labels), class_weight={0:1.0, 1:0.4})

loss,acc=vgg_conv.evaluate(test_data, test_labels, batch_size=16)
print('Loss and accuracy: ', loss,'&', acc)

pred = vgg_conv.predict(test_data, batch_size=16)
pred=np.argmax(pred, axis=-1)
labels=np.argmax(test_labels, axis=-1)
from sklearn.metrics import classification_report
print(classification_report(labels, pred))

cm = confusion_matrix(labels, pred)
sns.heatmap(cm, annot=True, fmt='g',xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])