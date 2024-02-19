import math
import re
import numpy as np
import nltk
from nltk.tokenize import LineTokenizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('wordnet')

K = [1, 3, 5, 7, 9, 11, 13, 15]
rep = ['binary', 'raw', 'tf_idf']
dist = ['manhattan', 'euclidian', 'cosine']


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


# define k-nearest neighbors classifier
class NearestNeighborClassifier:
    def __init__(self, k=None, bow_rep=None, dist_func=None):
        # List of possible words
        self.word_list = None
        self.x_train = None
        self.y_train = None

        # number of neighbors
        self.k = k
        # Bag of words representation mode (raw, binary, tf_idf)
        self.bow_rep = bow_rep
        # Distance function
        self.dist_func = dist_func

    # populate word list, fit training data
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        # create list of all possible words
        # represent each review as a sparse vector

    # predict test data based on training data
    def predict(self, test):
        pred = []
        # compare each test review against all training reviews

    # import text data, populate word list, train data
    def import_train(self, path):
        tk = LineTokenizer()
        wnl = nltk.wordnet.WordNetLemmatizer()
        file = open(path, 'r', encoding='utf-8')
        content = file.read()
        content = content.lower().replace('', '')
        # split each review into a separate line
        reviews = tk.tokenize(content)
        y = np.zeros(len(reviews))
        # clean text, remove stop words, stem words
        for i in range(len(reviews)):
            reviews[i] = reviews[i].replace('<br />', '').replace('#eof', '')
            y[i] = int(reviews[i][0:2])
            reviews[i] = reviews[i][2:]
            # remove all non-alphabetical characters (except for score)
            reviews[i] = re.sub('[^a-zA-Z]', ' ', reviews[i])
            # split each line into separate words
            reviews[i] = reviews[i].split()
            reviews[i] = [wnl.lemmatize(word) for word in reviews[i] if word not in stop]
            print(i)
        print(reviews[0])


# cross validation for parameters
# number of neighbors (k), bag-of-words representation (binary / raw frequency count / TF*IDF), distance/similarity
# measure (at least two)
# split train set into train and validation
# fit training data
# create output file based on test data
test = NearestNeighborClassifier()
test.import_train('train.txt')
