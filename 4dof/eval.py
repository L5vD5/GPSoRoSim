import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn import gaussian_process
import random
import json
import os

train_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/4dof/trainInterpolate.json')
with open(train_path, "r") as st_json:

    train_json = json.load(st_json)['data']

test_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/4dof/valExtrapolate.json')
with open(test_path, "r") as st_json:

    test_json = json.load(st_json)['data']
    
print('num of train data {}'.format(len(train_json)))
print('num of test data {}'.format(len(test_json)))

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

print('maximum train X {}'.format(np.max(train_X)))
print('maximum test X {}'.format(np.max(test_X)))
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.set_title('GT(Blue) and Pred(Red)')

train_X = train_X[0:100]
train_y = train_y[0:100]
GPR = gaussian_process.GaussianProcessRegressor(kernel=gaussian_process.kernels.RBF(length_scale=1.2))
GPR.fit(train_X, train_y.reshape(-1,27))

pred = GPR.predict(test_X)

def update(frame, data, line):
  line[0].set_data(data[frame, :, 0], data[frame, :, 1])
  line[0].set_3d_properties(data[frame, :, 2])

  return line

def update2(frame, data2, line2):
  line2[0].set_data(data2[frame, :, 0], data2[frame, :, 1])
  line2[0].set_3d_properties(data2[frame, :, 2])

  return line2


pred = pred.reshape(-1, 9, 3)
print(np.argmax((test_y - pred)/test_y, axis=0))
# pred += [0.1, 0, 0]

line = ax.plot(test_y[0, :, 0], test_y[0, :, 1], test_y[0, :, 2], '.-')
line2 = ax.plot(pred[0, :, 0], pred[0, :, 1], pred[0, :, 2], '.-', c='r')
# line = ax.plot(test_y[3956, :, 0], test_y[3956, :, 1], test_y[3956, :, 2], '.-')
# line2 = ax.plot(pred[3956, :, 0], pred[3956, :, 1], pred[3956, :, 2], '.-', c='r')

ani = FuncAnimation(fig, update, fargs=[test_y, line], frames=range(len(test_y)), interval=300)
ani2 = FuncAnimation(fig, update2, fargs=[pred, line2], frames=range(len(test_y)), interval=300)
plt.show()
