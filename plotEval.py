import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn import gaussian_process
import random

fig = plt.figure()
fig2 = plt.figure()

# data = np.load('data.npz')['arr_0']
train_data = np.loadtxt('data/data111.txt', delimiter=',')
# random_idx = random.sample(range(len(train_data)), 100)
# print(train_data)
# train_data = train_data[random_idx]
test_data = np.loadtxt('data/data333.txt', delimiter=',')
ax = fig.add_subplot(projection='3d')
ax2 = fig2.add_subplot(projection='3d', )
ax.set_title('GT')
ax2.set_title('Predicted')

input = train_data[:,:3]
# print(input.shape)
# input = random.sample(input, 100)
label = train_data[:,3:]
test_input = test_data[:,:3]
test_label = test_data[:,3:]
print(np.max(input))
# print("input: ", input)
# print("label: ", label)

GPR = gaussian_process.GaussianProcessRegressor()
GPR.fit(input, label)

# print(input[0].reshape(-1,3))
# print(np.linalg.norm(GPR.predict(input[0].reshape(-1,3))))
# print(np.linalg.norm(GPR.predict(input[0].reshape(-1,3))-label[0]))
print(input)
pred = GPR.predict(test_input.reshape(-1,3))
print(np.max(pred))
def update(frame, data, line):
  line[0].set_data(data[20*frame, 0:3*nMarker-2:3], data[20*frame, 1:3*nMarker-1:3])
  line[0].set_3d_properties(data[20*frame, 2:3*nMarker:3])

  return line

def update2(frame, data2, line2):
  line2[0].set_data(data2[20*frame, 0:3*nMarker-2:3], data2[20*frame, 1:3*nMarker-1:3])
  line2[0].set_3d_properties(data2[20*frame, 2:3*nMarker:3])

  return line2


ax.set_xlim3d([-10, 10])
ax.set_ylim3d([-10, 10])
ax.set_zlim3d([-10, 10])
ax2.set_xlim3d([-10, 10])
ax2.set_ylim3d([-10, 10])
ax2.set_zlim3d([-10, 10])

nMarker = 10

# label = label.reshape(-1,3)

# for i in range(nMarker):
#   plt.plot(pred[0:400:20,3*i], pred[0:400:20,3*i+1], pred[0:400:20,3*i+2], c='k')
#   plt.plot(test_label[0:400:20,3*i], test_label[0:400:20,3*i+1], test_label[0:400:20,3*i+2], c='b')

line = ax.plot(test_label[0, 0:3*nMarker-2:3], test_label[0, 1:3*nMarker-1:3], test_label[0, 2:3*nMarker:3], '.-')
line2 = ax2.plot(pred[0, 0:3*nMarker-2:3], pred[0, 1:3*nMarker-1:3], pred[0, 2:3*nMarker:3], '.-', c='r')

ani = FuncAnimation(fig, update, fargs=[test_label, line], frames=range(50), interval=100)
ani2 = FuncAnimation(fig2, update2, fargs=[pred, line2], frames=range(50), interval=100)
plt.show()
