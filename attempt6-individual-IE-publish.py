import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import string as s

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import FreqDist
from nltk.tokenize import word_tokenize

# stopwords = set(stopwords.words('english'))
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

class NaiveBayesText:
	def __init__(self, X, y, alpha=1, weighted=False):
		self.X = X
		self.y = y
		self.weighted = weighted
		self.alpha = alpha
		self.pers_types = dict()
		self.reversed_pers_types = dict()

		self.create_encode_dicts(self.y)
		# Еncode each pers type using pers_types dict
		self.y_encoded = np.array([self.pers_types[yi] for yi in self.y])

		print("Priors")
		unique, counts = np.unique(self.y_encoded, return_counts=True)
		print(np.asarray((unique, counts)).T) # Unequal distribution of priors, must weight classes in Naive Bayes

		self.preds = None
		self.decoded_preds = None
		self.scores = None
		self.model = None


	def create_encode_dicts(self, labels):
		labels_sorted = sorted(list(set(labels)))
		for i, lab in enumerate(labels_sorted):
			self.pers_types[lab] = i # Assigns each personality type a number

		for key, value in self.pers_types.items():
			self.reversed_pers_types[value] = key # Reversed dict

		print(self.pers_types)

	def train(self, use_val=False):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=1)

		# Now just training a model with given alpha
		vectorizer = TfidfVectorizer(max_features=20000) # Takes only most common 20k words
		vectors = vectorizer.fit_transform(X_train) # train
		vectors_test = vectorizer.transform(X_test) # test

		if self.weighted:
			unique, counts = np.unique(y_train, return_counts=True)
			priors = np.asarray((unique, counts)).T
			total = priors[0,1] + priors[1,1]
			# Weights are inverse of their relative incidence in priors 
			weights = np.where(y_train == priors[0,0], total/priors[0,1], total/priors[1,1])
			print('weight samples\n', priors)
			print(weights[:10])
		else:
			weights = None

		self.model = MultinomialNB(alpha=self.alpha)
		self.model.fit(vectors, y_train, sample_weight=weights)

		self.preds = self.model.predict(vectors_test).tolist()
		print("Predicts these categories:", set(self.preds))
		unique, counts = np.unique(self.preds, return_counts=True)
		print("With dist:", np.asarray((unique, counts)).T)

		self.scores = self.get_scores(y_test)
		print("\n\n")
		return self.scores


	def decode_y(self, yvals):
		y_decoded = []
		for y in yvals:
			y_decoded.append(self.reversed_pers_types[y])

		return y_decoded


	def get_scores(self, y_test):
		acc = accuracy_score(y_test, self.preds)
		pre = precision_score(y_test, self.preds, average='weighted')
		rec = recall_score(y_test, self.preds, average='weighted')

		return (acc, pre, rec)


df = pd.read_csv('mbti_1.csv')

# Split into individual comments
df['comment_list'] = df['posts'].apply(lambda x: x.split("|||"))

def preprocess(string):
	string = re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'lien', string).lower()
	string = string.strip('...')

	# Want "it's" to stay there, want "parent's" to become "parents"
	replace_dict = {':)': 'smiley', ':(': 'frowny', '..': ' ellipsis ', '%': ' % ', '-': ' - ', '.': ' . ', ';': ' ; ', ':': ' :', '?': ' ? ', '!': ' ! ', '\'': '', '/': ' / ', '(': ' ( ', ')': ' ) ', ',': ''}

	for word, punc in replace_dict.items():
		string = string.replace(word, punc)

	# Also want to turn all parent's to parent.
	# string = string.translate(str.maketrans('', '', s.punctuation))
	return ' '.join(string.split())

split = df.explode('comment_list').reset_index(drop=True).drop(['posts'], axis=1)

# Filter with no. of characters per comment > 50
split['comment_len'] = split['comment_list'].apply(lambda x: len(x))
split['comment_list'] = split['comment_list'].apply(lambda x: preprocess(x))
split = split[split['comment_len'] > 50]

split['IE'] = split['type'].apply(lambda x: x[0])
split['SN'] = split['type'].apply(lambda x: x[1])
split['TF'] = split['type'].apply(lambda x: x[2])
split['JP'] = split['type'].apply(lambda x: x[3])

dimensions = ['IE', 'SN', 'TF', 'JP']

portion = split.sample(frac=0.8)

# Cross-validate to find optimal alpha
CV_record = []
alphas = [1, 0.1, 0.01, 0.001, 0.0001]


for alpha in alphas:
	preds_by_dim = []
	for dim in dimensions:
		NB = NaiveBayesText(portion['comment_list'].values, portion[dim].values, alpha=alpha)
		scores = NB.train()
		CV_record.append((alpha, dim, scores))

		preds_by_dim.append(NB.preds)
		del NB

	# print('preds_by_dim', preds_by_dim)
	_, _, _, y_test_dec = train_test_split(portion['comment_list'], portion['type'].values, random_state=1)

	preds = [''.join([EI, SN, TF, JP]) for EI, SN, TF, JP in zip(preds_by_dim[0], preds_by_dim[1], preds_by_dim[2], preds_by_dim[3])]
	# print(preds)
	# print(y_test_dec)
	total = 0.0
	n_instances = len(preds)

	for i in range(n_instances):
		count = 0.0
		for ch in range(len(preds[i])):
			if preds[i][ch] == y_test_dec[i][ch]:
				count += 1
		total += count/4
	myscore = total/n_instances

	print(f'--------With alpha = {alpha}--------\nAccuracy: {scores[0]}\nPrecision: {scores[1]}\nRecall: {scores[2]}\nMyscore: {myscore}\n\n\n')

	for ch in range(len(preds[i])):
		count = 0.0
		for i in range(n_instances):
			if preds[i][ch] == y_test_dec[i][ch]:
				count+=1

		print(f'On {dimensions[ch]} dimension:\nAccuracy: {count/n_instances}')


print(CV_record)








