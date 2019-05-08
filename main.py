import numpy as np
from sklearn import *
import matplotlib.pyplot as plt


# Hàm vẽ biểu đồ classification
# src: https://gist.github.com/dennybritz/ff8e7c2954dd47a4ce5f


def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# Tạo dữ liệu
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)


# Khởi tạo cho giảm đạo hàm
learning_rate = 0.01


def sigmoid(z):
    """Trả về giá trị sau khi đi qua hàm sigmoid"""
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """Trả về giá trị softmax"""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def caculate_cost(model, X, y):
    """Trả về giá trị cost với model hiện tại"""

    # giá trị predict với X và model hiện tại
    a2 = predict(model, X)['output']

    # từ a2(nx2) và y(nx1) thực tế ta lấy ra được giá trị output cần thiết (nx1) để tối ưu
    # nếu y = 0 thì chọn output 0 còn y = 1 thì chọn output 1
    a2_fit = a2[range(len(X)), y]
    # cost function:
    # L = -ln(1-h(x))*(1-y) - y*ln(h(x))
    cost = - np.log(1 - a2_fit) * (1 - y) - np.log(a2_fit) * y

    return 1./len(y) * np.sum(cost)


def build_model(X, y, learning_rate=0.001, hidden_node_number=5, loop_number=2000):
    """return model"""

    # initial model
    # trọng số random w1 (2xhnn)
    W1 = np.random.randn(X.shape[1], hidden_node_number)
    # độ lệch bias random b1 (1xhnn)
    b1 = np.zeros((1, hidden_node_number))
    # trọng số random w2 (hnnx2)
    W2 = np.random.randn(hidden_node_number, 2)
    # độ lệch bias random b2 (1x2)
    b2 = np.zeros((1, 2))

    model = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

    for i in range(loop_number):
        # predict với model hiện tại
        a2 = predict(model, X)['output']

        # back propagation
        # delta2 (ở output layer) bằng ypredict - yreal (yreal gán = 1 do có 2 output)
        delta2 = a2
        delta2[range(len(X)), y] -= 1

        # delta1 = (W2.T) * delta2 * gradient(sigmoid(z1))
        #        = (W2.T) * delta2 * (a1*(1-a1))
        # trong đó a1 = sigmoid(z1) và z1 = X*W1+b1
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)

        # đạo hàm của trọng số bằng giá trị đầu vào nhân với delta đầu ra
        dW2 = (a1.T).dot(delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        delta1 = (delta2).dot(W2.T) * a1 * (1-a1)

        dW1 = (X.T).dot(delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Cập nhật model
        W1 += -learning_rate*dW1
        W2 += -learning_rate*dW2
        b1 += -learning_rate*db1
        b2 += -learning_rate*db2

        model = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
        print(caculate_cost(model, X, y))


    return model

def predict(model, X):
    """Trả về bộ nhãn dự đoán (0 hoặc 1) với bộ dữ liệu X đầu vào"""
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = softmax(z2)
    # a2 hiện tại là ma trận nx2 do 2 ouput mà y ban đầu là nx1 với giá trị là 0 hoặc 1
    # cần chuyển a2 thành nx1 bằng cách lựa chọn đầu ra nào lớn hơn (y = 0 ouput trên (0), y=1 ouput dưới (1))
    return {'label': np.argmax(a2, axis=1), 'output': a2}


model = build_model(X, y)

# Biểu đồ điểm dữ liệu
# plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# Vẽ biểu đồ với model
plot_decision_boundary(lambda X: predict(model, X)['label'])
plt.show()
