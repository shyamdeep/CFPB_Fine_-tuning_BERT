#import the libraries
import re
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from transformers import AutoTokenizer, TFBertModel
import numpy as np



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained('bert-base-uncased')


st.write("# Customer Complaints Classification")

complaint_text = st.text_input("Enter a complaint for classification")

max_len=128
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")


embeddings = bert(input_ids, attention_mask = input_mask)[0] # 0 = last hidden state, 1 = poller_output
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32, activation='relu')(out)

y = Dense(5, activation='softmax')(out)

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True
model.load_weights("bert-clf-weights")




def classify_complaint(model, complaint):
    predict_input = tokenizer(
        text=complaint_text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
        )
    predicted = model.predict({'input_ids': predict_input['input_ids'], 'attention_mask': predict_input['attention_mask']})
    probabilities = tf.nn.softmax(predicted).numpy()[0]
    label = np.argmax(probabilities)
    
    product_dict ={0:'credit_reporting', 1:'debt_collection', 2:'mortgages_and_loans',3:'credit_card',4:'retail_banking'}
    
      
    return {'label': product_dict[label], 'complaint_prob': probabilities[label]}






def predictor(text):
    predict_input = tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True
        )
    predicted = model.predict({'input_ids': predict_input['input_ids'], 'attention_mask': predict_input['attention_mask']})
    probas = tf.nn.softmax(predicted).numpy()
    return probas




if complaint_text != '':
    result = classify_complaint(model, complaint_text)
    st.write(result)
    
    explain_pred = st.button('Explain Predictions')
    if explain_pred:
        with st.spinner('Generating explanations'):
            class_names = ['credit_reporting','debt_collection','mortgages_and_loans','credit_card','retail_banking']
            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(complaint_text, predictor, num_features=5, num_samples=1,top_labels=2)
            exp.show_in_notebook(text=complaint_text)
            components.html(exp.as_html(), height=3500)

