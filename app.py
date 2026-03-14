import streamlit as st
from keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('model.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)

st.title('Movie Sentiment Analysis')

review = st.text_input('Enter your review')

if st.button('Predict Sentiment') and review.strip():

    sequence = tokenizer.texts_to_sequences([review])

    padded_sequence = pad_sequences(sequence, padding='post', maxlen=39)

    prediction = model.predict(padded_sequence)

    predicted_class = np.argmax(prediction, axis=1)[0]
    if predicted_class == 1:
       st.success("Positive 😊")
    else:
       st.error("Negative 😡")

    sentiment_map = {0:'Negative',1:'Positive',2:'Neutral'}


