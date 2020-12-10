#!/usr/bin/env python
# coding: utf-8

# ### Import the Libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import itertools
import collections
import pandas as pd
from plotly.offline import iplot,plot
from plotly.offline import init_notebook_mode
import cufflinks as cf
import seaborn as sns

init_notebook_mode(connected=True)
cf.go_offline()


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read Dataset

# In[3]:


#Dataset Path
path = 'database/'
window_size = 160
maximum_counting = 10000

X = list() #data
y = list() #labels


# In[4]:


df = pd.read_csv("database/100.csv")


# ### plotting data representation of a single patient

# In[5]:


df.iplot()


# <img src='images/patient1.png'/>

# In[7]:


# Read Filenames
filenames = next(os.walk(path))[2]

# Split and save .csv , .txt 
records = list()
annotations = list()
filenames.sort()

for f in filenames:
    filename, file_extension = os.path.splitext(f)
    
    # *.csv
    if(file_extension == '.csv'):
        records.append(path + filename + file_extension)

    # *.txt
    else:
        annotations.append(path + filename + file_extension)


# In[8]:


classes = ['N','L','R','A','V','/']
n_classes = len(classes)
count_classes = [0] * n_classes


# In[9]:


for r in range(0,len(records)):
    signals = []

    with open(records[r], 'rt') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|') # read CSV file\
        row_index = -1
        for row in spamreader:
            if(row_index >= 0):
                signals.insert(row_index, int(row[1]))
            row_index += 1

    # Read anotations: R position and Arrhythmia class
    with open(annotations[r], 'r') as fileID:
        data = fileID.readlines() 
        beat = list()

        for d in range(1, len(data)): # 0 index is Chart Head
            splitted = data[d].split(' ')
            splitted = filter(None, splitted)
            next(splitted) # Time... Clipping
            pos = int(next(splitted)) # Sample ID
            arrhythmia_type = next(splitted) # Type

            if(arrhythmia_type in classes):
                arrhythmia_index = classes.index(arrhythmia_type)
                if count_classes[arrhythmia_index] > maximum_counting: # avoid overfitting
                    pass
                else:
                    count_classes[arrhythmia_index] += 1
                    if(window_size < pos and pos < (len(signals) - window_size)):
                        beat = signals[pos-window_size+1:pos+window_size]
                        X.append(beat)
                        y.append(arrhythmia_index)


# ### Loading the data set in a dataframe

# In[10]:


data = X
columns = [x for x in range(1,320)]
df2 = pd.DataFrame(data = data, columns =columns)

df2.plot(legend = False)


# ### Applying the feature extractions and classifications

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA


# In[13]:


scaler1 = StandardScaler()
scaler1.fit(X)
feature_scaled = scaler1.transform(X)


# ### Data after scaling to be fitted in PCA

# In[14]:


plt.hist(feature_scaled,bins = 10,log = True)


# ### Applying PCA in scaled data

# In[15]:


model_pca = PCA(n_components = 6)
model_pca.fit(feature_scaled)
feature_scaled_pca = model_pca.transform(feature_scaled)
print("Shape of the scaled and PCA'ed Features: ", np.shape(feature_scaled_pca))


# In[16]:


data = feature_scaled_pca
columns = ['N','L','R','A','V','/']
scaled = pd.DataFrame(data = data,columns = columns)


# In[17]:


scaled.plot()


# In[18]:


scaled.iplot()


# In[19]:


scaled.plot.hist(log = True,legend  = False)


# ### Final scaled and PCA applied Data 

# In[20]:


sns.pairplot(scaled)


# In[21]:


feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))

print("Variance Ratio of the 6 Principal Components Analysis: ", feat_var_rat)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(feature_scaled_pca, y, test_size=0.33)

print("X_train : ", len(X_train))
print("X_test  : ", len(X_test))
print("y_train : ", collections.Counter(y_train))
print("y_test  : ", collections.Counter(y_test))


# In[23]:


from sklearn.svm import SVC 
C_value = 4
gamma_value = 'auto'
use_probability = False
multi_mode = 'ovo'

svm_model = SVC(C=C_value, kernel='rbf', degree=3, gamma=gamma_value,  
                    coef0=0.0, shrinking=True, probability=use_probability, tol=0.001, 
                    cache_size=200, verbose=False, 
                    max_iter=-1, decision_function_shape=multi_mode, random_state=None)
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test) 
  
# model accuracy for X_test   
accuracy_2 = svm_model.score(X_test, y_test) 
  
# creating a confusion matrix 
cm_2 = confusion_matrix(y_test, svm_predictions)


# In[24]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #cm[i, j] = 0 if np.isnan(cm[i, j]) else cm[i, j]
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print("\nTest Accuracy: {0:f}%\n".format(accuracy_2*100))
plot_confusion_matrix(cm_2, classes, normalize=True)


# In[25]:


report = classification_report(y_test, svm_predictions)
print(report)


# In[ ]:




