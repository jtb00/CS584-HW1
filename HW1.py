import math
import re
import time
import random
import numpy as np
import heapq
import nltk
from nltk.tokenize import LineTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('wordnet')

# representation of infinity
inf = 9999
# Number of neighbors
K = [1, 3, 5, 7, 9, 11, 13, 15]
# Bag of words representation
rep = ['binary', 'raw', 'tf_idf']
# Distance method
dist = ['manhattan', 'euclidean', 'cosine']
# Word list will track (list_size) most frequent words (inf = all words)
list_size = [100, 200, 500, 1000, 2000, 5000, inf]


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
    if x_norm == 0 or y_norm == 0:
        return inf
    return 1 - (dot / (x_norm * y_norm))


def binary_matrix(reviews, word_list):
    matrix = []
    for r in reviews:
        vector = []
        for w in word_list:
            if w in r:
                vector.append(1)
            else:
                vector.append(0)
        matrix.append(vector)
    return matrix


def raw_matrix(reviews, word_list):
    matrix = []
    for r in reviews:
        vector = np.zeros(len(word_list))
        w_list = list(word_list)
        for w in r:
            if w in w_list:
                ind = w_list.index(w)
                vector[ind] += 1
        matrix.append(vector)
    return matrix


# import training data, populate word list, return x and y matrices
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
        for val in reviews[i]:
            # if this word isn't in the word list, add it with count 1
            if val not in word_list.keys():
                word_list[val] = 1
            # otherwise, add 1 to its count
            else:
                word_list[val] += 1
        # print(i)
    if num_words != inf:
        freq_words = heapq.nlargest(num_words, word_list, key=word_list.get)
        word_list = freq_words
    # represent each review as a sparse vector
    x_train = []
    count = 0
    if bow_rep == 'binary':
        x_train = binary_matrix(reviews, word_list)
    elif bow_rep == 'tf_idf':
        b_matrix = np.matrix(binary_matrix(reviews, word_list))
        r_matrix = raw_matrix(reviews, word_list)
        # tracks number of reviews containing each word
        num_reviews_list = np.sum(b_matrix, axis=0)
        idf_list = np.array(np.log(len(reviews) / num_reviews_list))
        x_train = []
        for i in range(len(r_matrix)):
            vector = []
            for j in range(len(r_matrix[i])):
                if r_matrix[i][j] != 0:
                    vector.append(r_matrix[i][j] / len(reviews[i]) * idf_list[0][j])
                else:
                    vector.append(0)
            x_train.append(vector)
    else:
        x_train = raw_matrix(reviews, word_list)
    return x_train, y_train, word_list


# import test data, return x matrix
def import_test(path, word_list, bow_rep):
    tk = LineTokenizer()
    wnl = nltk.wordnet.WordNetLemmatizer()
    file = open(path, 'r', encoding='utf-8')
    content = file.read()
    content = content.lower().replace('', '')
    # split each review into a separate line
    reviews = tk.tokenize(content)
    # clean text, remove stop words, stem words
    for i in range(len(reviews)):
        reviews[i] = reviews[i].replace('<br />', '').replace('#eof', '')
        # remove all non-alphabetical characters
        reviews[i] = re.sub('[^a-zA-Z]', ' ', reviews[i])
        # split each line into separate words
        reviews[i] = reviews[i].split()
        reviews[i] = [wnl.lemmatize(word) for word in reviews[i] if word not in stop]
        #filter out words not in the word list
        reviews[i] = [word for word in reviews[i] if word in word_list]
    x_test = []
    count = 0
    if bow_rep == 'binary':
        x_test = binary_matrix(reviews, word_list)
    elif bow_rep == 'tf_idf':
        b_matrix = np.matrix(binary_matrix(reviews, word_list))
        r_matrix = raw_matrix(reviews, word_list)
        # tracks number of reviews containing each word
        num_reviews_list = np.sum(b_matrix, axis=0)
        idf_list = np.array(np.log(len(reviews) / num_reviews_list))
        x_test = []
        for i in range(len(r_matrix)):
            vector = []
            for j in range(len(r_matrix[i])):
                if r_matrix[i][j] != 0:
                    vector.append(r_matrix[i][j] / len(reviews[i]) * idf_list[0][j])
                else:
                    vector.append(0)
            x_test.append(vector)
    else:
        x_test = raw_matrix(reviews, word_list)
    return x_test


def accuracy(pred, act):
    count = 0
    for i in range(len(pred)):
        if pred[i] == act[i]:
            count += 1
    return count / len(pred)


# create output file based on test data
def write_to_file(data):
    f = open("result.txt", "w")
    for i in range(len(data)):
        if data[i] == 1:
            f.write("+1")
        else:
            f.write("-1")
        if i != (len(data) - 1):
            f.write("\n")
    f.close()


def random_cross_val(x_train, y_train, words, bow_rep, iter):
    best_k = 0
    best_d = ''
    best_score = 0
    random.seed(0)
    for i in range(iter):
        k = K[random.randrange(0, len(K))]
        d = dist[random.randrange(0, len(dist))]
        nnc = NearestNeighborClassifier(words, k, bow_rep, d, len(words))
        start = (i % 10) * 2500
        end = (i % 10) * 2500 + 2500
        # split train set into train and validation
        x_val = x_train[start:end]
        y_val = y_train[start:end]
        if start == 0:
            x_tr = x_train[end:]
            y_tr = y_train[end:]
        elif end == 25000:
            x_tr = x_train[0:start]
            y_tr = y_train[0:start]
        else:
            x_tr = np.concatenate((x_train[0:start], x_train[end:]))
            y_tr = np.concatenate((y_train[0:start], y_train[end:]))
        print("dist = " + d + ", K = " + str(k))
        nnc.fit(x_tr, y_tr)
        tic = time.time()
        pred = nnc.predict(x_val)
        toc = time.time()
        print("Elapsed: " + str(toc - tic) +" sec.")
        score = accuracy(pred, y_val)
        if score > best_score:
            best_score = score
            best_k = k
            best_d = d
        print("Slice " + str(i + 1) + ": Accuracy = " + str(score))
    return best_score, best_k, best_d


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
        count = 0
        # compare each test review against all training reviews
        for t in test:
            distances = [inf] * self.k
            labels = [0] * self.k
            if self.dist_func == 'manhattan':
                for i in range(len(self.x_train)):
                    dist = manhattan_dist(t, self.x_train[i])
                    if dist < min(distances):
                        ind = distances.index(max(distances))
                        distances[ind] = dist
                        labels[ind] = self.y_train[i]
            elif self.dist_func == 'euclidean':
                for i in range(len(self.x_train)):
                    dist = euclidean_dist(t, self.x_train[i])
                    if dist < min(distances):
                        ind = distances.index(max(distances))
                        distances[ind] = dist
                        labels[ind] = self.y_train[i]
            else:
                for i in range(len(self.x_train)):
                    dist = cosine_dist(t, self.x_train[i])
                    if dist < min(distances):
                        ind = distances.index(max(distances))
                        distances[ind] = dist
                        labels[ind] = self.y_train[i]
            mean = 0
            for l in labels:
                mean += l
            mean = mean / len(labels)
            if mean > 0:
                pred.append(1)
            else:
                pred.append(-1)
            # print(count)
            count += 1
        return pred


# cross validation for parameters
num_words = inf
best_score = 0
best_k = 0
best_d = ''
best_rep = ''
final_x_train = []
final_y_train = []
final_words = {}
for i in range(3):
    x_train, y_train, words = import_train('train.txt', num_words, rep[i])
    score, k, d = random_cross_val(x_train, y_train, words, rep[i], 10)
    if score > best_score:
        best_score = score
        best_k = k
        best_d = d
        best_rep = rep[i]
        final_x_train = x_train
        final_y_train = y_train
        final_words = words

x_test = import_test('test.txt', final_words, best_rep)
nnc = NearestNeighborClassifier(final_words, best_k, best_rep, best_d, len(final_words))
pred = nnc.predict(x_test)
write_to_file(pred)
