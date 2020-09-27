#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning Inception V3 using Keras

# Please download the dataset from the below url

# In[1]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt


# In[4]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'G:/kaggle/Tomato pata/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_path = 'G:/kaggle/Tomato pata/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid'


# In[5]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# In[6]:


# don't train existing weights
for layer in inception.layers:
    layer.trainable = False


# In[7]:


# useful for getting number of output classes
folders = glob('G:/kaggle/Tomato pata/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/*')


# In[8]:


# our layers - you can add more if you want
x = Flatten()(inception.output)


# In[9]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# In[10]:



# view the structure of the model
model.summary()


# In[11]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[12]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[13]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 4,
                                                 class_mode = 'categorical')


# In[14]:


test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (224, 224),
                                            batch_size = 4,
                                            class_mode = 'categorical')


# In[ ]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set), 
  validation_steps=len(test_set)
)


# In[15]:




# Place tensors on the CPU
with tf.device('/GPU:0'):
    r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=10,
    steps_per_epoch=len(training_set), 
    validation_steps=len(test_set)
)
    


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[26]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('G:/kaggle/Tomato pata/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/model_inception.h5')


# In[ ]:





# In[34]:



y_pred = model.predict(test_set)


# In[28]:


y_pred


# In[29]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[30]:


y_pred


# In[ ]:





# In[31]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[38]:


model=load_model('G:/kaggle/Tomato pata/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/model_inception.h5')


# In[40]:


img_data


# In[42]:


img=image.load_img('C:/Users/Boishakhi/Downloads/download.jpg',target_size=(224,224))


# In[43]:


x=image.img_to_array(img)
x


# In[44]:


x.shape


# In[45]:


x=x/255


# In[46]:


import numpy as np
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[47]:


model.predict(img_data)


# In[48]:


a=np.argmax(model.predict(img_data), axis=1)


# In[49]:


a==1


# In[50]:


import tensorflow as tf


# In[51]:


tf.__version__


# In[ ]:




