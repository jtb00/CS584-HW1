import math
import re
import numpy as np
import heapq
import nltk
from nltk.tokenize import LineTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('wordnet')

# Number of neighbors
K = [1, 3, 5, 7, 9, 11, 13, 15]
# Bag of words representation
rep = ['binary', 'raw', 'tf_idf']
# Distance method
dist = ['manhattan', 'euclidian', 'cosine']
# Word list will track (list_size) most frequent words (9999 = all words)
list_size = [100, 200, 500, 1000, 2000, 5000, 9999]


# calculate Manhattan distance between vectors x and y
def manhattan_dist(x, y):
    result = 0
    for i in range(0, len(x)):
        result = result + abs(x[i] - y[i])
    return result


# calculate Euclidean distance between vectors x and y
def euclidean_dist(x, y):
    result = 0
    for i in range(0, len(x)):
        result = result + (x[i] - y[i]) ** 2
    return math.sqrt(result)


# calculate Cosine distance between vectors x and y
def cosine_dist(x, y):
    dot = 0
    x_norm = 0
    y_norm = 0
    for i in range(0, len(x)):
        dot = dot + x[i] * y[i]
        x_norm = x_norm + x[i] ** 2
        y_norm = y_norm + y[i] ** 2
    x_norm = math.sqrt(x_norm)
    y_norm = math.sqrt(y_norm)
    return 1 - (dot / (x_norm * y_norm))


# import text data, populate word list, train data
def import_train(path, num_words, bow_rep):
    tk = LineTokenizer()
    wnl = nltk.wordnet.WordNetLemmatizer()
    file = open(path, 'r', encoding='utf-8')
    content = file.read()
    content = content.lower().replace('', '')
    # split each review into a separate line
    reviews = tk.tokenize(content)
    y_train = np.zeros(len(reviews))
    word_list = {}
    # clean text, remove stop words, stem words
    for i in range(len(reviews)):
        reviews[i] = reviews[i].replace('<br />', '').replace('#eof', '')
        y_train[i] = int(reviews[i][0:2])
        reviews[i] = reviews[i][2:]
        # remove all non-alphabetical characters (except for score)
        reviews[i] = re.sub('[^a-zA-Z]', ' ', reviews[i])
        # split each line into separate words
        reviews[i] = reviews[i].split()
        reviews[i] = [wnl.lemmatize(word) for word in reviews[i] if word not in stop]
        # populate word list
        for w in reviews[i]:
            # if this word isn't in the word list, add it with count 1
            if w not in word_list.keys():
                word_list[w] = 1
            # otherwise, add 1 to its count
            else:
                word_list[w] += 1
        print(i)
    if num_words != 9999:
        freq_words = heapq.nlargest(num_words, word_list, key=word_list.get)
        word_list = freq_words
    # represent each review as a sparse vector
    x_train = []
    # count = 0
    for r in reviews:
        # finish this later
        if bow_rep == 'tf_idf':
            vector = np.zeros(len(word_list))
            w_list = list(word_list)
            w_count = np.zeros(len(r))
            w_total = len(r)
            for w in r:
                w_total += 1
        elif bow_rep == 'raw':
            vector = np.zeros(len(word_list))
            w_list = list(word_list)
            for w in r:
                ind = w_list.index(w)
                vector[ind] += 1
        # rep = binary, default case
        else:
            vector = []
            for w in word_list:
                if w in r:
                    vector.append(1)
                else:
                    vector.append(0)
            # print(count)
            # count += 1
        x_train.append(vector)
    return x_train, y_train, word_list


# define k-nearest neighbors classifier
class NearestNeighborClassifier:
    def __init__(self, word_list=None, k=None, bow_rep=None, dist_func=None, num_words=100):
        # List of possible words
        self.word_list = word_list
        self.x_train = x_train
        self.y_train = y_train

        # number of neighbors
        self.k = k
        # Bag of words representation mode (raw, binary, tf_idf)
        self.bow_rep = bow_rep
        # Distance function
        self.dist_func = dist_func
        self.num_words = num_words

    # populate word list, fit training data
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # predict test data based on training data
    def predict(self, test):
        pred = []
        # compare each test review against all training reviews
        for t in test:



x_train, y_train, words = import_train('train.txt', 100, 'binary')
# cross validation for parameters
for i in range(5):
    start = i * 5000
    end = i * 5000 + 5000
    # split train set into train and validation
    x_val = x_train[start:end]
    x_tr = x_train[0:start] + x_train[end:]
    y_val = y_train[start:end]
    y_tr = y_train[0:start] + y_train[end:]
    for d in dist:
        for k in K:
            nnc = NearestNeighborClassifier(words, k, 'binary', d, 100)
            nnc.fit(x_tr, y_tr)
            # pred = nnc.predict(x_val)
# create output file based on test data
