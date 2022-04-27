import sklearn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn import gaussian_process

fig = plt.figure()
# data = np.load('data.npz')['arr_0']
train_data = np.loadtxt('data/data111.txt', delimiter=',')

ax = fig.add_subplot(projection='3d')
ax.set_title('plot 3d')

input = train_data[:,:3]
label = train_data[:,3:]
# print("input: ", input)
# print("label: ", label)

GPR = gaussian_process.GaussianProcessRegressor()
GPR.fit(input, label)

# print(input[0].reshape(-1,3))
# print(np.linalg.norm(GPR.predict(input[0].reshape(-1,3))))
# print(np.linalg.norm(GPR.predict(input[0].reshape(-1,3))-label[0]))

pred = GPR.predict(input.reshape(-1,3))

def update(frame, data, data2, line):
  line[0].set_data(data[frame*20, 0:3*nMarker-2:3], data[frame*20, 1:3*nMarker-1:3])
  line[0].set_3d_properties(label[frame*20, 2:3*nMarker:3])
#   line2[0].set_data(data2[frame*20, 0:3*nMarker-2:3], data2[frame*20, 1:3*nMarker-1:3])
#   line2[0].set_3d_properties(label[frame*20, 2:3*nMarker:3])

  # line2.set_data(data2[frame, :, 0], data2[frame, :, 1])
  # line2.set_3d_properties(data2[frame, :, 2])
  return line


# ax.set_xlim3d([-0.5, 0.5])
# ax.set_ylim3d([-0.5, 0.5])
# ax.set_zlim3d([-0.5, 0.5])

nMarker = 10
# label = label.reshape(-1,3)

for i in range(nMarker):
  plt.plot(pred[0:400:20,3*i], pred[0:400:20,3*i+1], pred[0:400:20,3*i+2], c='k')
  plt.plot(label[0:400:20,3*i], label[0:400:20,3*i+1], label[0:400:20,3*i+2], c='b')

line = ax.plot(pred[0, 0:3*nMarker-2:3], pred[0, 1:3*nMarker-1:3], pred[0, 2:3*nMarker:3], '.-')
# line2 = ax.plot(label[0, 0:3*nMarker-2:3], label[0, 1:3*nMarker-1:3], label[0, 2:3*nMarker:3], '.-', c='r')

ani = FuncAnimation(fig, update, fargs=[pred, label, line], frames=range(20), interval=1000)
plt.show()
