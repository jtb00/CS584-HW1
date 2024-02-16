import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = stopwords.words('english')

K = [1, 3, 5, 7, 9, 11, 13, 15]
rep = ['binary', 'raw', 'tf_idf']
dist = ['manhattan', 'euclidian', 'cosine']


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

    # represent each review as a sparse vector
    # fit training data
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # predict test data based on training data
    def predict(self, test):
        pred = []
        # compare each test review against all training reviews

    # import text data
    def import_data(self, path):
        file = open(path, 'r')
        # clean text, remove stop words, stem words
        # create list of all possible words

# cross validation for parameters
    # number of neighbors (k), bag-of-words representation (binary / raw frequency count / TF*IDF), distance/similarity
    # measure (at least two)
    # split train set into train and validation
# fit training data
# create output file based on test data
