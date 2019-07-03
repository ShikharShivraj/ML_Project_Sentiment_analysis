from string import punctuation, digits
import numpy as np
import random


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    # Finds the hinge loss on a single data point given specific classification parameters.
    if label * (np.dot(feature_vector, theta) + theta_0) > 0:
        return 0
    else:
        return 1 - label * (np.dot(feature_vector, theta) + theta_0)


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    t = np.arange(1000)
    sgn = np.arange(1000)
    sum = 0
    for i in range(len(feature_matrix)):
        sum = sum + hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
    return sum / len(feature_matrix)


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    # Properly updates the classification parameter, theta and theta_0, on a
    # single step of the perceptron algorithm.
    sgn = label * (np.dot(current_theta, feature_vector) + current_theta_0)
    if sgn > 2 ** -30:
        return current_theta, current_theta_0;
    else:
        return current_theta + label * feature_vector, current_theta_0 + label;


def perceptron(feature_matrix, labels, T):
    n = feature_matrix[0].shape
    theta = np.zeros(n, )
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            pass
    return theta, theta_0


def average_perceptron(feature_matrix, labels, T):
    n = feature_matrix[0].shape
    theta = np.zeros(n, )
    theta_0 = 0
    c = 0
    sum_theta = np.zeros(n, )
    sum_theta_0 = 0
    # k=n[0]*T
    for t in range(T):
        c = 0
        for i in get_order(feature_matrix.shape[0]):
            # Your code here
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            sum_theta += theta
            sum_theta_0 += theta_0
            c = c + 1
    return sum_theta / (c * T), sum_theta_0 / (c * T)


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return current_theta * (1 - eta * L) + eta * label * feature_vector, current_theta_0 + eta * label
    else:
        return current_theta * (1 - eta * L), current_theta_0


def pegasos(feature_matrix, labels, T, L):
    # For each update, set learning rate = 1/sqrt(t),
    # where t is a counter for the number of updates performed so far (between 1
    # and nT inclusive).
    n = feature_matrix[0].shape
    theta = np.zeros(n, )
    theta_0 = 0
    m = 1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1 / np.sqrt(m)
            m = m + 1
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
            pass
    return theta, theta_0


def classify(feature_matrix, theta, theta_0):

    # A classification function that uses theta and theta_0 to classify a set of
    # data points.
    k = len(feature_matrix)
    a = np.empty(k, )
    for i in range(len(feature_matrix)):
        if (np.dot(feature_matrix[i], theta) + theta_0) > 2 ** -30:
            a[i] = 1
        else:
            a[i] = -1
    return a


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    # Trains a linear classifier and computes accuracy.
    # The classifier is trained on the train data. The classifier's
    # accuracy on the train and validation data is then returned.
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    a = classify(train_feature_matrix, theta, theta_0)
    training_accuracy = accuracy(a, train_labels)
    b = classify(val_feature_matrix, theta, theta_0)
    validation_accuracy = accuracy(b, val_labels)
    return training_accuracy, validation_accuracy


def extract_words(input_string):

    # Helper function for bag_of_words()
    # Inputs a text string
    # Returns a list of lowercase words in the string.
    # Punctuation and digits are separated out into their own words.

    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    # Inputs a list of string reviews
    # Returns a dictionary of unique unigrams occurring over the input
    stopwords = {'i': 0, 'me': 1, 'my': 2, 'myself': 3, 'we': 4, 'our': 5, 'ours': 6, 'ourselves': 7, 'you': 8,
                 'your': 9, 'yours': 10, 'yourself': 11, 'yourselves': 12, 'he': 13, 'him': 14, 'his': 15,
                 'himself': 16, 'she': 17, 'her': 18, 'hers': 19, 'herself': 20, 'it': 21, 'its': 22, 'itself': 23,
                 'they': 24, 'them': 25, 'their': 26, 'theirs': 27, 'themselves': 28, 'what': 29, 'which': 30,
                 'who': 31, 'whom': 32, 'this': 33, 'that': 34, 'these': 35, 'those': 36, 'am': 37, 'is': 38, 'are': 39,
                 'was': 40, 'were': 41, 'be': 42, 'been': 43, 'being': 44, 'have': 45, 'has': 46, 'had': 47,
                 'having': 48, 'do': 49, 'does': 50, 'did': 51, 'doing': 52, 'a': 53, 'an': 54, 'the': 55, 'and': 56,
                 'but': 57, 'if': 58, 'or': 59, 'because': 60, 'as': 61, 'until': 62, 'while': 63, 'of': 64, 'at': 65,
                 'by': 66, 'for': 67, 'with': 68, 'about': 69, 'against': 70, 'between': 71, 'into': 72, 'through': 73,
                 'during': 74, 'before': 75, 'after': 76, 'above': 77, 'below': 78, 'to': 79, 'from': 80, 'up': 81,
                 'down': 82, 'in': 83, 'out': 84, 'on': 85, 'off': 86, 'over': 87, 'under': 88, 'again': 89,
                 'further': 90, 'then': 91, 'once': 92, 'here': 93, 'there': 94, 'when': 95, 'where': 96, 'why': 97,
                 'how': 98, 'all': 99, 'any': 100, 'both': 101, 'each': 102, 'few': 103, 'more': 104, 'most': 105,
                 'other': 106, 'some': 107, 'such': 108, 'no': 109, 'nor': 110, 'not': 111, 'only': 112, 'own': 113,
                 'same': 114, 'so': 115, 'than': 116, 'too': 117, 'very': 118, 's': 119, 't': 120, 'can': 121,
                 'will': 122, 'just': 123, 'don': 124, 'should': 125, 'now': 126}
    dictionary = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stopwords:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    # Returns the bag-of-words feature matrix representation of the data.
    # The returned matrix is of shape (n, m), where n is the number of reviews
    # and m the total number of entries in the dictionary.

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                # feature_matrix[i, dictionary[word]] = feature_matrix[i, dictionary[word]] + 1 -- count feature
                feature_matrix[i, dictionary[word]] = 1
    return feature_matrix


def accuracy(preds, targets):

    # returns the percentage and number of correct predictions.
    return (preds == targets).mean()

