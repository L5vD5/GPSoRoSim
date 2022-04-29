import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import json

train_path = '/home/yoonbyung/Dev/GPSoRoSim/data/4dof/trainInterpolate.json'
with open(train_path, "r") as st_json:

    train_json = json.load(st_json)['data']

test_path = '/home/yoonbyung/Dev/GPSoRoSim/data/4dof/valExtrapolate.json'
with open(test_path, "r") as st_json:

    test_json = json.load(st_json)['data']

train_X = np.empty(shape=(0,4))
train_y = np.empty(shape=(0,9,3))
for data in train_json:
  train_X = np.concatenate((train_X, [data['actuation']]), axis=0)
  train_y = np.concatenate((train_y, [data['position']]), axis=0)

test_X = np.empty(shape=(0,4))
test_y = np.empty(shape=(0,9,3))
for data in test_json:
  test_X = np.concatenate((test_X, [data['actuation']]), axis=0)
  test_y = np.concatenate((test_y, [data['position']]), axis=0)

fig = plt.figure()
ax = fig.add_subplot()

train_X = train_X[0:100]
train_y = train_y[0:100]

X = np.concatenate((train_X, test_X))
y = np.concatenate((train_y, test_y))

X_embedded = TSNE(n_components=2,
                   init='random').fit_transform(X)
print(X_embedded)

ax.scatter(X_embedded[0:100,0], X_embedded[0:100,1], c='r')
ax.scatter(X_embedded[101:,0], X_embedded[101:,1], s=0.1)
plt.show()
