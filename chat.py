import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import urllib
import pickle
import random
import url 

with open("/Users/vipul1/Documents/GitHub/CHATBOT/intents.json") as file:
    data = json.load(file)

with open("/Users/vipul1/Documents/GitHub/CHATBOT/resource.json") as file:
    resource = json.load(file)

model1 = keras.models.load_model('/Users/vipul1/Documents/GitHub/CHATBOT/chat_model.h5')
model = keras.models.load_model('/Users/vipul1/Documents/GitHub/CHATBOT/chat_model1.h5')
#     # load tokenizer object
with open('/Users/vipul1/Documents/GitHub/CHATBOT/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('/Users/vipul1/Documents/GitHub/CHATBOT/tokenizer1.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)
#     # load label encoder object
with open('/Users/vipul1/Documents/GitHub/CHATBOT/label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)
with open('/Users/vipul1/Documents/GitHub/CHATBOT/label_encoder1.pickle', 'rb') as enc:
     lbl_encoder1 = pickle.load(enc)

def chat(inp):
        max_len = 20
        respo=""
        respo1=""
        respo2=""
        result= model1.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                respo= np.random.choice(i['responses'])
                if tag=='demand':
                    result1 = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer1.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
                    tag1 = lbl_encoder1.inverse_transform([np.argmax(result1)])
                    
                    for j in resources['intents']:
                      if j['tag']==tag1:
                        respo= np.random.choice(j['responses'])

                    
             
        
        return respo
print(chat(input("phrase:")))
