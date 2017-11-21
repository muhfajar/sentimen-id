import json
import sys

import nltk
import random

import numpy as np
import tflearn
import tensorflow as tf

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

with open('data/kata.json') as json_data:
    data = json.load(json_data)

categories = list(data.keys())

words = []
docs = []

for y, each_category in enumerate(data.keys()):
    for x, each_sentence in enumerate(data[each_category]):
        each_sentence = stemmer.stem(each_sentence)
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        docs.append((w, each_category))
        sys.stdout.write('\r')
        sys.stdout.write("Processing {} of {}  [{} / {}]".format(y + 1, len(data.keys()), x + 1, len(data[each_category])))
        sys.stdout.flush()
    print("")

# new line
print("\n")

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

training = []
output = []
output_empty = [0] * len(categories)

for doc in docs:
    bow = []
    token_words = doc[0]
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

tf.reset_default_graph()
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs', checkpoint_path='tflearn_model/checkpoint')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True, snapshot_epoch=True, snapshot_step=500)
model.save('tflearn_model/model.tflearn')

# TODO: load if model exist
model.load("tflearn_model/model.tfl")

sent_1 = input("Keyword: ")


def get_tf_record(sentence):
    global words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    bow = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return np.array(bow)


print(categories[np.argmax(model.predict([get_tf_record(sent_1)]))])
