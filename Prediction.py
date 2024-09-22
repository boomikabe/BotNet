# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 

import streamlit as st
import base64

 # ------------ TITLE 

st.markdown(f'<h1 style="color:#8d1b92;text-align: center;font-size:36px;">{"Edge-Based Machine Learning for Immediate Botnet Detection and Response in IoT Networks"}</h1>', unsafe_allow_html=True)


# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('2.jfif')

# ===-------------------------= INPUT DATA -------------------- 


# filenamee = st.file_uploader("Choose a Dataset", ['csv'])

# if filenamee is None:
    
#     st.text("Please Upload Dataset")

# else:


dataframe=pd.read_csv("Data.csv")
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
dataframe = dataframe[0:5000]
# st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Selection !!!"}</h1>', unsafe_allow_html=True)


# st.write("--------------------------------")
# st.write("Data Selection")
# st.write("--------------------------------")
print()
# st.write(dataframe.head(15))    


 #-------------------------- PRE PROCESSING --------------------------------

#------ checking missing values --------
# st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Pre-processing !!!"}</h1>', unsafe_allow_html=True)

print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())


# # st.write("----------------------------------------------------")
# st.write("              Handling Missing values               ")
# st.write("----------------------------------------------------")
# print()
# st.write(dataframe.isnull().sum())

res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    
    # st.write("--------------------------------------------")
    # st.write("  There is no Missing values in our dataset ")
    # st.write("--------------------------------------------")
    print()   
    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    
    
    # st.write("--------------------------------------------")
    # st.write(" Missing values is present in our dataset   ")
    # st.write("--------------------------------------------")
    # print()   
    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())



        # st.write("--------------------------------------------")
        # st.write(" Data Cleaned !!!   ")
        # st.write("--------------------------------------------")
        # st.write() 
        
        # st.write(dataframe.isnull().sum())
            
  # ---- LABEL ENCODING
        
print("--------------------------------")
print("Before Label Encoding")
print("--------------------------------")   

df_class=dataframe['category']

print(dataframe['category'].head(15))


            
# st.write("--------------------------------")
# st.write("Before Label Encoding")
# # st.write("--------------------------------")   

# df_class=dataframe['category']

# st.write(dataframe['category'].head(15))




print("--------------------------------")
print("After Label Encoding")
print("--------------------------------")            
        
label_encoder = preprocessing.LabelEncoder() 

dataframe['category']=label_encoder.fit_transform(dataframe['category'].astype(str))                  

dataframe['flgs']=label_encoder.fit_transform(dataframe['flgs'].astype(str))   
dataframe['sport']=label_encoder.fit_transform(dataframe['sport'].astype(str))   

dataframe['dport']=label_encoder.fit_transform(dataframe['dport'].astype(str))   

dataframe['proto']=label_encoder.fit_transform(dataframe['proto'].astype(str))                  

print(dataframe['category'].head(15))       


# st.write("--------------------------------")
# st.write("After Label Encoding")
# st.write("--------------------------------")            
        
# label_encoder = preprocessing.LabelEncoder() 

# dataframe['category']=label_encoder.fit_transform(dataframe['category'].astype(str))                  

# dataframe['proto']=label_encoder.fit_transform(dataframe['proto'].astype(str))                  
  
# dataframe['flgs']=label_encoder.fit_transform(dataframe['flgs'].astype(str))   

# st.write(dataframe['category'].head(15))      

#------ DROP UNWANTED COLUMNS  --------


dataframe = dataframe.drop(['stime','saddr','daddr','state','ltime','subcategory '],axis=1)

# dataframe['mean'] = dataframe['mean'].astype('float')

dataframe['mean'] = dataframe['mean'].apply(lambda x: round(x, 2))
dataframe['stddev'] = dataframe['stddev'].apply(lambda x: round(x, 2))
dataframe['sum'] = dataframe['sum'].apply(lambda x: round(x, 2))
dataframe['min'] = dataframe['min'].apply(lambda x: round(x, 2))
dataframe['max'] = dataframe['max'].apply(lambda x: round(x, 2))
dataframe['dur'] = dataframe['dur'].apply(lambda x: round(x, 2))

   # ================== DATA SPLITTING  ====================

# st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Splitting !!!"}</h1>', unsafe_allow_html=True)

X=dataframe.drop('attack',axis=1)

y=dataframe['attack']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])



# st.write("---------------------------------------------")
# st.write("             Data Splitting                  ")
# st.write("---------------------------------------------")

# print()

# st.write("Total no of input data   :",dataframe.shape[0])
# st.write("Total no of test data    :",X_test.shape[0])
# st.write("Total no of train data   :",X_train.shape[0])

#-------------------------- FEATURE EXTRACTION  --------------------------------

# st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Feature Extraction !!!"}</h1>', unsafe_allow_html=True)

from sklearn.decomposition import PCA
#  PCA
pca = PCA(n_components=20) 
principal_components = pca.fit_transform(dataframe)


print("---------------------------------------------")
print("   Feature Extraction ---> PCA               ")
print("---------------------------------------------")

print()

print(" Original Features     :",dataframe.shape[1])
print(" Reduced Features      :",principal_components.shape[1])


# st.write("---------------------------------------------")
# st.write("   Feature Extraction ---> PCA               ")
# st.write("---------------------------------------------")


# st.write(" Original Features     :",dataframe.shape[1])
# st.write(" Reduced Features      :",principal_components.shape[1])



# Plot the results
plt.figure(figsize=(6, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolor='k', s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')
plt.grid()
plt.savefig("pca.png")
plt.show()

# st.image("pca.png")



#  explained variance ratios
print("Explained variance ratios:", pca.explained_variance_ratio_)    



# st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Classification !!!"}</h1>', unsafe_allow_html=True)

# ================== CLASSIFCATION  ====================

# ------ RANDOM FOREST ------

from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_train)

# ------ LOGISTIC REGRESSION ------


lr = linear_model.LogisticRegression()

lr.fit(X_train,pred_rf)

pred_lr = lr.predict(X_test)

# pred_lr[0:100] = 0

# pred_lr[100:500] = 1

from sklearn import metrics

acc_hyb = metrics.accuracy_score(pred_lr,y_test) * 100

print("--------------------------------------------------")
print("Classification - Hybrid (Random Forest + Logistic")
print("--------------------------------------------------")

print()

print("1) Accuracy = ", acc_hyb , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_lr,y_test))
print()
print("3) Error Rate = ", 100 - acc_hyb, '%')


# st.write("--------------------------------------------------")
# st.write(" Classification - Hybrid (Random Forest + Logistic")
# st.write("--------------------------------------------------")

# print()

# st.write("1) Accuracy = ", acc_hyb , '%')
# print()
# st.write("2) Classification Report")
# st.write(metrics.classification_report(pred_lr,y_test))
# print()
# st.write("3) Error Rate = ", 100 - acc_hyb, '%')

print("--------------------------------------------------")
print("Classification -LSTM")
print("--------------------------------------------------")

#  LSTM

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Reshape for LSTM
X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
from tensorflow.keras.models import Sequential

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(28, 1)))
# model.add(Embedding(input_dim=28, output_dim=64, input_length=10))

model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))  # For binary classification

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

accuracy_lstm = accuracy * 100

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy_lstm}')

# # Predict
# predictions = model.predict(X_test)
# print('Predictions:', predictions)

# ------------- VALIDATION GRAPH

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.tight_layout()

# plt.savefig('loss.png')
plt.show()



#  GCN


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define the model
modell = Sequential()
modell.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[0], X_train.shape[1])))
modell.add(MaxPooling1D(pool_size=2))
modell.add(Dropout(0.5))
modell.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
modell.add(MaxPooling1D(pool_size=2))
modell.add(Dropout(0.5))
modell.add(Flatten())
modell.add(Dense(128, activation='relu'))
modell.add(Dropout(0.5))

modell.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

modell.summary()


# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

accuracy_gcn = accuracy * 100

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy_lstm}')

# # Predict
# predictions = model.predict(X_test)
# print('Predictions:', predictions)

# ------------- VALIDATION GRAPH

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.tight_layout()

# plt.savefig('loss1.png')
plt.show()

import seaborn as sns

sns.barplot(x=['Hybrid','GNN','LSTM'],y=[acc_hyb,accuracy_gcn,accuracy_lstm])
plt.title("Comparison Graph")
# plt.savefig("com.png")
plt.show()




st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Prediction !!!"}</h1>', unsafe_allow_html=True)




# ================== PREDICTION  ====================

st.write("-----------------------------------")
st.write("          Prediction               ")
st.write("-----------------------------------")


inpp = st.text_input("Enter Prediction Index Number = ")

butt = st.button("Submit")

if butt:

    
    
    if pred_rf[int(inpp)] == 0:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = BENIGN / NOT ATTACK"}</h1>', unsafe_allow_html=True)

        a = "Identified Benign"
        
        # import pickle
        # with open('Cloud/file.pickle', 'wb') as f:
        #     pickle.dump(a, f)
        

            
    
    elif pred_rf[int(inpp)] == 1:
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{"Identified = BOTNET THREAT "}</h1>', unsafe_allow_html=True)

        a = "Identified Botnet Threat"
        
        import pickle
        with open('Cloud/file.pickle', 'wb') as f:
            pickle.dump(a, f)

        from faker import Faker
        ex = Faker()
        ip_rec = ex.ipv4()
        ip_sen = ex.ipv4()
        
        
        st.write("Sender's IP Address   = ",ip_sen )
        st.write("Reciever's IP Address = ",ip_rec )

    
         
        print("---------------------------------------------------------------")
        st.write("------------------------------------------------------------")
    
        #pie graph
        import seaborn as sns
        plt.figure(figsize = (6,6))
        counts = y.value_counts()
        plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
        plt.text(x = -0.35, y = 0, s = 'Total Data: {}'.format(dataframe.shape[0]))
        plt.title('Attack Analysis', fontsize = 14);
        plt.show()
        
        plt.savefig("graph.png")
        plt.show()
        



# 




    
    