import json
from sklearn import svm
import numpy as np
import cPickle as pickle

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def featurize(data):
    for d in data:
        feat = {}
        ings = d[u'ingredients']
        for ing in ings:
            for word in ing.strip().split(' '):
                word = word.lower()
                feat[word] = 1.
        d['feat'] = feat

train = load_data('data/train.json')
test = load_data('data/test.json')

featurize(train)
featurize(test)

ing_dict = {}
ing_count = 0

for d in train:
    for key in d['feat']:
        if key not in ing_dict:
            ing_dict[key] = ing_count
            ing_count += 1

cuisine_dict = {}
cuisine_count = 0
for d in train:
    cuisine = str(d[u'cuisine'])
    if cuisine not in cuisine_dict:
        cuisine_dict[cuisine] = cuisine_count
        cuisine_count += 1

cuisine_idict = {}
for k, v in cuisine_dict.items():
    cuisine_idict[v] = k

# question: training distribution != test distribution, if words in test not present in train.
print '#ingredient features', ing_count

def encode(data):
    y = []
    x = []
    for d in data:
        if u'cuisine' in d:
            cuisine = str(d[u'cuisine'])
            d['y'] = cuisine_dict[cuisine]
            y.append(d['y'])
        d['x'] = np.zeros(ing_count)
        for f, v in d['feat'].items():
            if f not in ing_dict:
                continue
            d['x'][ing_dict[f]] = v
        x.append(d['x'])
    return (np.array(x), np.array(y))

(trainX, trainY) = encode(train)
(testX, testY) = encode(test)

lin_clf = svm.LinearSVC()
lin_clf.fit(trainX, trainY)
predY = lin_clf.predict(testX)

output = open('result/submission.csv', 'w')
output.write('id,cuisine\n')
for (di, d) in enumerate(test):
    output.write('%d,%s\n' % (d[u'id'], cuisine_idict[predY[di]]))
output.close()

