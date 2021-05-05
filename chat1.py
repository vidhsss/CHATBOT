import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
from twe import twi 


from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import urllib

import random
from tornado.escape import linkify


with open("/Users/vipul1/Documents/GitHub/CHATBOT/intents.json") as file:
    intents= json.load(file)

# with open("/Users/vipul1/Documents/GitHub/CHATBOT/resource.json") as file:
#     resources = json.load(file)
# with open('/Users/vipul1/Documents/GitHub/CHATBOT/tokenizer.pickle', 'rb') as handle:
#     tokenizer1 = pickle.load(handle)
# with open('/Users/vipul1/Documents/GitHub/CHATBOT/label_encoder.pickle', 'rb') as enc:
#     lbl_encoder1 = pickle.load(enc)

words = pickle.load(open('/Users/vipul1/Documents/GitHub/CHATBOT/words.pkl','rb'))
classes = pickle.load(open('/Users/vipul1/Documents/GitHub/CHATBOT/classes.pkl','rb'))
# model_r= load_model('/Users/vipul1/Documents/GitHub/CHATBOT/chatbot_model_r.h5')
max_len=20
def cityy(sentence): 
    cities=['Andhra Pradesh',' Assam',' Arunachal Pradesh',' Bihar',' Goa',' Gujarat',' Jammu and Kashmir',' Jharkhand',' West Bengal',' Karnataka',' Kerala',' Madhya Pradesh',' Maharashtra',' Manipur',' Meghalaya',' Mizoram',' Nagaland',' Orissa',' Punjab',' Rajasthan',' Sikkim',' Tamil Nadu',' Tripura',' Uttaranchal',' Uttar Pradesh',' Haryana',' Himachal Pradesh','  Chhattisgarh','andhra pradesh',' assam',' arunachal pradesh',' bihar',' goa',' gujarat',' jammu and kashmir',' jharkhand',' west bengal',' karnataka',' kerala',' madhya pradesh',' maharashtra',' manipur',' meghalaya',' mizoram',' nagaland',' orissa',' punjab',' rajasthan',' sikkim',' tamil nadu',' tripura',' uttaranchal',' uttar pradesh',' haryana',' himachal pradesh','chhattisgarh']
    for city in cities: 
        if city in sentence: 
            return city
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
def chatbot_response(text):
    res3="."
    res1=""
    res=""
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    t=ints[0]['intent']
    tags=['oxygen','beds','plasma','medicine','ambulance']
    for tag in tags : 
        if t==tag: 
            city=cityy(text)
            if city==None: 
                city="delhi"
            search_words = tag+" available verified"+city 
            res3=twi(search_words)
            
            if res3==None: 
                search_words = tag+ " available" +city
                res3=twi(search_words)
            if res3==None: 
                res3=" no tweet found"
            res3=str("Most recent twitter search: "+res3)
            
        # result1 = model_r.predict(keras.preprocessing.sequence.pad_sequences(tokenizer1.texts_to_sequences([text]),
        #                                      truncating='post', maxlen=max_len))
        # tag1 = lbl_encoder1.inverse_transform([np.argmax(result1)])
                    
        # for j in resources['intents']:
        #     if j['tag']==tag1:
        # \\\\search_words = "#"+tag+"#available #delhi #verified"
        # res3=twi(search_words)
        #res2=(np.random.choice(j['responses']))
        # res3="\nMost recent verified twitter search:\n"+ res3
    # res="Please go through site: "+res   
    res1=str(res+"\n"+ res3)
    return linkify(res1)
print(chatbot_response(input("phrase")))
