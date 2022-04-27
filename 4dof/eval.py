import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn import gaussian_process
import random
import json

train_path = '/home/yoonbyung/Dev/GPSoRoSim/data/4dof/trainInterpolate.json'
with open(train_path, "r") as st_json:

    train_json = json.load(st_json)['data']

test_path = '/home/yoonbyung/Dev/GPSoRoSim/data/4dof/valInterpolate.json'
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
# fig2 = plt.figure()

ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.set_title('GT')
# ax2.set_title('Predicted')

# print(test_X.shape)
# print(test_y.shape)
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


# ax.set_xlim3d([-10, 10])
# ax.set_ylim3d([-10, 10])
# ax.set_zlim3d([-10, 10])
# ax2.set_xlim3d([-10, 10])
# ax2.set_ylim3d([-10, 10])
# ax2.set_zlim3d([-10, 10])
pred = pred.reshape(-1, 9, 3)
# print(np.argmax((test_y - pred)/test_y))
# print(test_y.shape)
pred += [0.1, 0, 0]

line = ax.plot(test_y[0, :, 0], test_y[0, :, 1], test_y[0, :, 2], '.-')
line2 = ax.plot(pred[0, :, 0], pred[0, :, 1], pred[0, :, 2], '.-', c='r')
# line = ax.plot(test_y[1455, :, 0], test_y[1455, :, 1], test_y[1595, :, 2], '.-')
# line2 = ax.plot(pred[1455, :, 0], pred[1455, :, 1], pred[1455, :, 2], '.-', c='r')

ani = FuncAnimation(fig, update, fargs=[test_y, line], frames=range(len(test_y)), interval=100)
ani2 = FuncAnimation(fig, update2, fargs=[pred, line2], frames=range(len(test_y)), interval=100)
plt.show()
