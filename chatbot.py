import nltk
import numpy as numpy
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


f=open('chat.txt','r',errors = 'ignore')

raw = f.read()
raw = raw.lower()

sent_tokens = nltk.sent_tokenize(raw) #sentence to list 
word_tokens = nltk.word_tokenize(raw)	# to list of words


#print(sent_tokens[:2])
#print(word_tokens[:2])

lemmer = nltk.stem.WordNetLemmatizer()

def Lemtokens(tokens):
	return[lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
	return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))



greet_inputs = ("hello","hi", "greetings", "sup", "what's up", "hey",)

greet_res = ["hi" "hey", "*nods*", "hi there", "hello", "I am glad! You are taking to me"]

def greeting(sentence):
	for word in sentence.split():
		if word.lower() in greet_inputs:
			return random.choice(greet_res)



def responce(user_response):
	robo_response=''
	sent_tokens.append(user_response)

	TfidfVec =TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
	tfidf = TfidfVec.fit_transform(sent_tokens)
	vals = cosine_similarity(tfidf[-1],tfidf)
	idx=vals.argsort()[0][-2]
	flat = vals.flatten()
	flat.sort()
	req_tfidf = flat[-2]

	if(req_tfidf==0):
		robo_response=robo_response+ "Iam sorry! I don't understand you."
		return robo_response

	else: 
		robo_response = robo_response+sent_tokens[idx]
		return robo_response

flag=True
print("ROBO: My name is Rishab. I will answer your queries about chatbots.\nIf you want to exit type Bye!")

while(flag==True):
	user_response=input("You: ")
	user_response=user_response.lower()
	if(user_response!='bye'):
		if user_response == 'thanks' or user_response == 'thank you':
			flag=False
			print("ROBO: You are welcome...")

		else:
			if(greeting(user_response)!=None):
				print("ROBO: "+greeting(user_response))
			else:
				print("ROBO: ",end="")
				print(responce(user_response))
				sent_tokens.remove(user_response)
	else:
		flag=False
		print("ROBO: Bye take care...")


