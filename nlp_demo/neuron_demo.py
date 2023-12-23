# see https://zhuanlan.zhihu.com/p/377513272 
import numpy as np

def sigmoid(x):
  # 我们的激活函数: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(self, inputs):
    # 权重乘以输入，与偏移量相加，然后通过激活函数
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994


# ... code from previous section here

class OurNeuralNetwork:
  '''
  神经网络:
    - 2 个输入
    - 1 个隐藏层，2 个神经元 (h1, h2)
    - 1 个输出层，1 个神经元 (o1)
  所有神经元有同样的权重和偏移量:
    - w = [0, 1]
    - b = 0
  '''
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    # 这里的 Neuron 类是上一节的代码里定义的
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # o1 的输入就是 h1 和 h2 的输出
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) # 0.7216325609518421


def mse_loss(y_true, y_pred):
  # y_true 和 y_pred 是长度相同的 numpy 数组.
  return ((y_true - y_pred) ** 2).mean()

y_true = np.array([1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred)) # 0.5