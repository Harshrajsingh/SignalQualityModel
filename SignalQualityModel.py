#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv("Signal.csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[6]:


Percentage_Missing = data.isnull().sum()* 100 / len(data)
missing_value_df =pd.DataFrame({'column_name': data.columns, 'percent_missing' : Percentage_Missing})
missing_value_df


# In[7]:


data.duplicated()


# In[8]:


data.duplicated().sum()


# In[9]:


data.loc[data.duplicated(), :]


# In[10]:


data.drop_duplicates(inplace=True)


# In[11]:


data.shape


# In[12]:


import seaborn as sns


# In[13]:


sns.displot(data, x ="Signal_Strength")


# In[14]:


print(data.Signal_Strength.value_counts())


# In[15]:


X = data.drop("Signal_Strength" , axis=1)
y = data.pop("Signal_Strength")


# In[16]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[17]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[18]:


from sklearn import preprocessing 
normalizer = preprocessing.Normalizer()
normalized_train_X = normalizer.fit_transform(X_train)
normalized_train_X


# In[19]:


normalized_test_X = normalizer.transform(X_test)
normalized_test_X


# In[21]:


import keras
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


# In[26]:


# Convert to "one-hot" vectors using the to_categorical function
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test_cat=to_categorical(y_test,num_classes)


# In[27]:


print("First 5 training lables as one-hot encoded vectors:\n", y_train[:5])


# In[87]:


from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
image_size=11


# In[88]:


# create model
model = Sequential() 


# In[89]:


model.add(Dense(256, activation='relu',kernel_initializer='he_uniform',input_shape=(image_size,))) ###Multiple Dense units with Relu activation
model.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))

model.add(Dense(num_classes, activation='softmax')) ### For multiclass classification Softmax is used 


# In[90]:


# Compile model
#RMS_prop=optimizers.RMSprop()   ## we can similarly use different optimizers like RMSprop, Adagrad and SGD 
adam = optimizers.Adam(lr=1e-3)
model.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy']) ### Loss function = Categorical cross entropy


# In[91]:


## Looking into our base model
model.summary()


# In[92]:


checkpoint = ModelCheckpoint("model_weights.h5",monitor='val_accuracy',
                            save_weights_only=True, mode='max',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,min_lr=0.00001,model='auto')

callbacks = [checkpoint,reduce_lr]


# In[93]:


# Fit the model
history=model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2,callbacks=callbacks)


# In[94]:


y_pred=model.predict(X_test)


# In[95]:


y_pred[0]


# In[96]:


y_pred_final=[]
for i in y_pred:
  y_pred_final.append(np.argmax(i))


# In[97]:


for i in y_pred_final:
    print(y_pred_final[i])


# In[98]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_final))


# In[99]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


cm=confusion_matrix(y_test,y_pred_final)

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


# In[100]:


loss, accuracy  = model.evaluate(normalized_test_X, y_test_cat, verbose=False)
    
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


# In[74]:


#Model 2


# In[75]:


# define model

from tensorflow.keras import losses
from tensorflow.keras import optimizers
from keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


# In[76]:


image_size=11

# create model
model_1 = Sequential()  

model_1.add(Dense(256, activation='relu',kernel_initializer='he_uniform',input_shape=(image_size,))) ###Multiple Dense units with Relu activation
model_1.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))
model_1.add(Dense(64, activation='relu',kernel_initializer='he_uniform'))## Weight Initialization
model_1.add(Dense(32, activation='relu',kernel_initializer='he_uniform'))
model_1.add(Dropout(0.5))
model_1.add(BatchNormalization())
model_1.add(Dense(num_classes, activation='softmax')) ### For multiclass classification Softmax is used


# In[77]:


# Compile model
adam = optimizers.Adam(lr=1e-3)
model_1.compile(loss=losses.categorical_crossentropy, optimizer=adam, metrics=['accuracy']) ### Loss function = Categorical cross entropy


# In[78]:


model_1.summary()


# In[79]:


checkpoint = ModelCheckpoint("model_weights_1.h5",monitor='val_accuracy',
                            save_weights_only=True, model='max',verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,min_lr=0.00001,model='auto')

callbacks = [checkpoint,reduce_lr]


# In[80]:


# Fit the model
history=model_1.fit(normalized_train_X, y_train, validation_split=0.2, epochs=10, batch_size=128, verbose=2, callbacks=callbacks)


# In[81]:


y_pred_1=model_1.predict(normalized_test_X)


# In[82]:


y_pred_final_1=[]
for i in y_pred_1:
  y_pred_final_1.append(np.argmax(i))


# In[83]:


for i in y_pred_final_1:
    print(y_pred_final_1[i])


# In[84]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_final_1))


# In[85]:


from sklearn.metrics import confusion_matrix
cm_1=confusion_matrix(y_test,y_pred_final_1)


# In[86]:


plt.figure(figsize=(10,7))
sns.heatmap(cm_1,annot=True,fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:




