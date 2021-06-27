from LinearNN import *
import matplotlib.pyplot as plt
from Loss import *

N_x = 2
N_y = 1
m = 10

Layer = [100]
N = len(Layer) +1

LNN1 = LinearNN(N_x,N_y,Layer)
LNN0 = LinearNN(N_x,N_y,[])
LNN1.compile()
LNN0.compile()
LNN0.set_weight(LNN1.summary.a * LNN1.weight_product_DP())
LNN0.summary.a = 1

LNN1.learning_rate = 0.0005
LNN0.learning_rate = (N / N_y) * LNN1.learning_rate

loss1 = []
loss0 = []

matrix_list = []

np.random.seed(0)
matrix_list.append(np.random.rand(N_x,m))
matrix_list.append(np.random.rand(N_y,m))


LNN1.forward(matrix_list[0],matrix_list[1])
LNN0.forward(matrix_list[0],matrix_list[1])

for x in range(100):
    LNN1.forward(matrix_list[0],matrix_list[1])
    loss1.append(LNN1.Loss())
    LNN1.backward()


for x in range(100):
    LNN0.forward(matrix_list[0],matrix_list[1])
    loss0.append(LNN0.Loss())
    LNN0.backward()
print(loss1)
print(loss0)


plt.plot(loss1, label='Deep Linear Network loss')
plt.plot(loss0, label='Convex Problem loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


