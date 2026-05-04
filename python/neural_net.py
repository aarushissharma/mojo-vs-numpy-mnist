import numpy as np
import time
from tensorflow.keras.datasets import mnist

# loading & pre-processing data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255.0 # flattening + normalizing data
x_test  = x_test.reshape(-1, 784) / 255.0

def one_hot(y, classes=10):
  out = np.zeros((y.size, classes))
  out[np.arange(y.size), y] = 1
  return out
y_train_oh = one_hot(y_train)
y_test_oh = one_hot(y_test)


# initializing weights (2 layers)
np.random.seed(42)

W1 = np.random.randn(784,128) * 0.01 # 784 inputs and 128 neurons so 784x128 total weights
b1 = np.zeros((1, 128)) # 1 bias per neuron (initialized to 0)

W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

# activation functions
def relu(z): #ReLU (rectified linear unit): if + valye then keep itm if - then zero it
  return np.maximum(0, z)


def relu_deriv(z): # use during backprop
  return (z > 0).astype(float)

def softmax(z): # takes 10 raw scores & converts tham to 10 probabilities that sum to 1
  e = np.exp(z - np.max(z, axis=1, keepdims=True))
  return e / e.sum(axis=1, keepdims=True)


def forward(X):
  z1 = X @ W1 + b1
  a1 = relu(z1)
  z2 = a1 @ W2 + b2
  a2 = softmax(z2)
  return z1, a1, z2, a2

def cross_entropy(a2, y): # how wrong predictions are 
  return -np.mean(np.sum(y * np.log(a2 + 1e-8), axis=1))

# backprop
def backward(X, y, z1, a1, a2, lr=0.1): # determines how much each weight contribited to error
  global W1, b1, W2, b2
  m = X.shape[0]

  dz2 = a2 - y # gradient of the loss w respect to output layer
  dW2 = a1.T @ dz2 / m
  db2 = dz2.mean(axis=0, keepdims=True)

  da1 = dz2 @ W2.T
  dz1 = da1 * relu_deriv(z1)
  dW1 = X.T @ dz1 / m 
  db1 = dz1.mean(axis=0, keepdims=True)

  W1 -= lr * dW1
  b1 -= lr * db1
  W2 -= lr * dW2
  b2 -= lr * db2

# training loop
epochs = 20
batch_size = 64

for epoch in range(epochs):
  idx = np.random.permutation(x_train.shape[0])
  x_s, y_s = x_train[idx], y_train_oh[idx]

  for i in range(0, x_train.shape[0], batch_size):
    xb = x_s[i:i+batch_size]
    yb = y_s[i:i+batch_size]
    z1, a1, z2, a2 = forward(xb)
    backward(xb, yb, z1, a1, a2)

  _, _, _, a2_full = forward(x_train)
  loss = cross_entropy(a2_full, y_train_oh)
  preds = np.argmax(a2_full, axis=1)
  acc = np.mean(preds == y_train)
  print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Acc: {acc:.4f}")

runs = 100
start = time.perf_counter()

for _ in range(runs):
  forward(x_test)
end = time.perf_counter()
avg_ms = (end - start) / runs * 1000
print(f"Avg forward pass time over {runs} runs: {avg_ms:.2f} ms")