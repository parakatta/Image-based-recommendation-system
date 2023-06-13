import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
import tensorflow as tf
from keras.applications.resnet import ResNet50, preprocess_input 
from keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors 
from PIL import Image

with open('./file.pkl', 'rb') as file :
    features = pickle.load(file)

def extract_features_from_img(img_path, model) : 
    img = imread(img_path)
    img = cv2.resize(img, (224,224))
    img = np.array(img)
    img_expanded = np.expand_dims(img, axis = 0)
    preprocessed = preprocess_input(img_expanded)
    result = model.predict(preprocessed).flatten()
    normalised = result / norm(result)
    return normalised

def predict_feature(features_img, features):
    neighbours = NearestNeighbors(n_neighbors = 5, algorithm = 'brute',metric = 'euclidean' )
    neighbours.fit(features)
    distance, indices = neighbours.kneighbors([features_img])
    return indices
    
def display_recommendations(indices):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.image(filenames[indices[0][0]])
    with col2:
        st.image(filenames[indices[0][1]])
    with col3:
        st.image(filenames[indices[0][2]])
    with col4:
        st.image(filenames[indices[0][3]])
    with col5:
        st.image(filenames[indices[0][4]])

dataset = pd.read_csv('./fashion.csv')
filenames = dataset['ImageURL'].to_list()

model = tf.keras.models.load_model('./model_file.h5')

#streamlit application
st.title("Image Based Recommendation system - Fashion dataset")

st.write('''
          The objective of the project was to create a system that could suggest visually 
          similar fashion items based on an input image.  
          *Method Used* : K Nearest Neighbors (KNN)  
    
         ''')
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file:
    img = Image.open(uploaded_file)
    resized_img = img.resize((200,200))
    st.image(resized_img)
    features_img = extract_features_from_img(uploaded_file, model)
    indices = predict_feature(features_img, features)
    display_recommendations(indices)
    
            
    
st.write("Try these..")
ufile = './images/ex1.jpg'
img = Image.open(ufile)
resize_img = img.resize((200,200))
st.image(resize_img)
if st.button('Recommend 1'):
        features_img = extract_features_from_img(ufile, model)
        indices = predict_feature(features_img, features)
        display_recommendations(indices)
ufile = './images/ex2.jpg'
img = Image.open(ufile)
resize_img = img.resize((200,200))
st.image(resize_img)
if st.button('Recommend 2'):
    features_img = extract_features_from_img(ufile, model)
    indices = predict_feature(features_img, features)
    display_recommendations(indices)
ufile = './images/ex3.jpg'
img = Image.open(ufile)
resize_img = img.resize((200,200))
st.image(resize_img)
if st.button('Recommend 3'):
    features_img = extract_features_from_img(ufile, model)
    indices = predict_feature(features_img, features)
    display_recommendations(indices)
ufile = './images/ex4.jpg'
img = Image.open(ufile)
resize_img = img.resize((200,200))
st.image(resize_img)
if st.button('Recommend 4'):
    features_img = extract_features_from_img(ufile, model)
    indices = predict_feature(features_img, features)
    display_recommendations(indices)
        