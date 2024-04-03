import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def derivative_tanh(x):
    return 1 - np.power(x, 2)


class generalize_data:
    @staticmethod
    def linear(n):
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)

    @staticmethod
    def xor():
        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)
            if 0.1*i == 0.5:
                continue
            inputs.append([0.1*i, 1-0.1*i])
            labels.append(1)
        print(inputs)
        return np.array(inputs), np.array(labels).reshape(21, 1)


class net:
    def __init__(self, hidden_size, num_step):
        self.hidden_size = hidden_size
        self.num_step = num_step
        self.EPSILON = 1e-5

        self.W = [None, np.random.randn(hidden_size[0], 2), np.random.randn(
            hidden_size[1], hidden_size[0]), np.random.randn(1, hidden_size[1])]
        self.b = [None, np.zeros((hidden_size[0], 1)), np.zeros(
            (hidden_size[1], 1)), np.zeros((1, 1))]
        self.z = [None, np.zeros((hidden_size[0], 1)), np.zeros(
            (hidden_size[1], 1)), np.zeros((1, 1))]
        self.a = [None, np.zeros((hidden_size[0], 1)), np.zeros(
            (hidden_size[1], 1)), np.zeros((1, 1))]

    def train(self, X, y):
        learning_rate = [0.5]
        for i in range(len(learning_rate)):
            loss_value = []
            square_grad_w1, square_grad_w2, square_grad_w3, grad_squared_sum_b1, grad_squared_sum_b2, grad_squared_sum_b3 = 0, 0, 0, 0, 0, 0
            for ep in range(self.num_step):
                pred_y = self.forward(X)
                square_grad_w1, square_grad_w2, square_grad_w3, grad_squared_sum_b1, grad_squared_sum_b2, grad_squared_sum_b3 = self.backward(
                    y, X, learning_rate[i], square_grad_w1, square_grad_w2, square_grad_w3, grad_squared_sum_b1, grad_squared_sum_b2, grad_squared_sum_b3)
                loss = self.compute_cost(y)
                loss_value.append(loss)
                acc = (1.0-np.sum(np.abs(y-np.round(pred_y)))/y.shape[1])*100
                print(f'Epochs {ep}: loss={loss:.5f} accuracy={acc:.2f}%')
            # different learning rate
            time_step = list(range(len(loss_value)))
        #     plt.plot(time_step, loss_value, label=learning_rate[i])
        # plt.legend(loc='best', title='learning rate')
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.title('loss curve')
        # plt.grid(True)
        # plt.show()
        return time_step, loss_value

        # learning curve
        # time_step = list(range(len(loss_value)))
        # plt.plot(time_step, loss_value, 'b')
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.title('loss curve')
        # plt.grid(True)
        # plt.show()

    def forward(self, inputs):

        self.z[1] = np.dot(self.W[1], inputs) + self.b[1]
        self.a[1] = relu(self.z[1])

        self.z[2] = np.dot(self.W[2], self.a[1]) + self.b[2]
        self.a[2] = tanh(self.z[2])

        self.z[3] = np.dot(self.W[3], self.a[2]) + self.b[3]
        self.a[3] = sigmoid(self.z[3])
        return self.a[3]

    def compute_cost(self, ground_true):
        m = ground_true.shape[1]

        loss = (1/m) * (-np.dot(ground_true,
                                np.log(self.a[3]).T+self.EPSILON) - np.dot((1-ground_true), np.log(1-self.a[3]+self.EPSILON).T))
        return float(loss)

    def backward(self, ground_true, X, learning_rate, grad_squared_sum_w1, grad_squared_sum_w2, grad_squared_sum_w3, grad_squared_sum_b1, grad_squared_sum_b2, grad_squared_sum_b3):
        batch_size = ground_true.shape[1]

        gradient_a3 = - \
            (np.divide(ground_true, self.a[3]+self.EPSILON)) + \
            np.divide(1-ground_true, 1-self.a[3]+self.EPSILON)
        gradient_z3 = gradient_a3 * derivative_sigmoid(self.a[3])
        gradient_w3 = (1/batch_size) * np.dot(gradient_z3, self.a[2].T)
        gradient_b3 = (1/batch_size) * \
            np.sum(gradient_z3, axis=1, keepdims=True)

        gradient_a2 = np.dot(self.W[3].T, gradient_z3)
        gradient_z2 = gradient_a2 * derivative_tanh(self.a[2])
        gradient_w2 = (1/batch_size) * np.dot(gradient_z2, self.a[1].T)
        gradient_b2 = (1/batch_size) * \
            np.sum(gradient_z2, axis=1, keepdims=True)

        gradient_a1 = np.dot(self.W[2].T, gradient_z2)
        gradient_z1 = gradient_a1 * derivative_relu(self.a[1])
        gradient_w1 = (1/batch_size) * np.dot(gradient_z1, X.T)
        gradient_b1 = (1/batch_size) * \
            np.sum(gradient_z1, axis=1, keepdims=True)

        gradient_w = [0, gradient_w1, gradient_w2, gradient_w3]
        gradient_b = [0, gradient_b1, gradient_b2, gradient_b3]

        optimizer = 'adam'

        if optimizer == 'GD':
            self.W[1] -= learning_rate * gradient_w1
            self.W[2] -= learning_rate * gradient_w2
            self.W[3] -= learning_rate * gradient_w3
            self.b[1] -= learning_rate * gradient_b1
            self.b[2] -= learning_rate * gradient_b2
            self.b[3] -= learning_rate * gradient_b3
        elif optimizer == 'momentum':
            vt = np.zeros_like(self.W[1])
            self.W[1] += (0.9 * vt - learning_rate * gradient_w1)
            vt = np.zeros_like(self.W[2])
            self.W[2] += (0.9 * vt - learning_rate * gradient_w2)
            vt = np.zeros_like(self.W[3])
            self.W[3] += (0.9 * vt - learning_rate * gradient_w3)
            vt = np.zeros_like(self.b[1])
            self.b[1] += (0.9 * vt - learning_rate * gradient_b1)
            vt = np.zeros_like(self.b[2])
            self.b[2] += (0.9 * vt - learning_rate * gradient_b2)
            vt = np.zeros_like(self.b[3])
            self.b[3] += (0.9 * vt - learning_rate * gradient_b3)
        elif optimizer == 'adagrad':
            grad_squared_sum_w1 += gradient_w1**2
            grad_squared_sum_w2 += gradient_w2**2
            grad_squared_sum_w3 += gradient_w3**2
            adaptive_lr_w1 = learning_rate / \
                (np.sqrt(grad_squared_sum_w1) + self.EPSILON)
            adaptive_lr_w2 = learning_rate / \
                (np.sqrt(grad_squared_sum_w2) + self.EPSILON)
            adaptive_lr_w3 = learning_rate / \
                (np.sqrt(grad_squared_sum_w3) + self.EPSILON)
            self.W[1] -= adaptive_lr_w1 * gradient_w1
            self.W[2] -= adaptive_lr_w2 * gradient_w2
            self.W[3] -= adaptive_lr_w3 * gradient_w3
            grad_squared_sum_b1 += gradient_b1**2
            grad_squared_sum_b2 += gradient_b2**2
            grad_squared_sum_b3 += gradient_b3**2
            adaptive_lr_b1 = learning_rate / \
                (np.sqrt(grad_squared_sum_b1) + self.EPSILON)
            adaptive_lr_b2 = learning_rate / \
                (np.sqrt(grad_squared_sum_b2) + self.EPSILON)
            adaptive_lr_b3 = learning_rate / \
                (np.sqrt(grad_squared_sum_b3) + self.EPSILON)
            self.b[1] -= adaptive_lr_b1 * gradient_b1
            self.b[2] -= adaptive_lr_b2 * gradient_b2
            self.b[3] -= adaptive_lr_b3 * gradient_b3
            return grad_squared_sum_w1, grad_squared_sum_w2, grad_squared_sum_w3, grad_squared_sum_b1, grad_squared_sum_b2, grad_squared_sum_b3
        elif optimizer == 'adam':
            learning_rate = 0.001
            for i in range(1, 4):
                m = np.zeros_like(self.W[i])
                v = np.zeros_like(self.W[i])
                m = 0.9 * m + (1 - 0.9) * gradient_w[i]
                v = 0.999 * v + (1 - 0.999) * (gradient_w[i] ** 2)
                m_hat = m / (1 - 0.9)
                v_hat = v / (1 - 0.999)
                self.W[i] = self.W[i] - learning_rate * \
                    m_hat / (np.sqrt(v_hat) + self.EPSILON)
            for j in range(1, 4):
                m = np.zeros_like(self.b[j])
                v = np.zeros_like(self.b[j])
                m = 0.9 * m + (1 - 0.9) * gradient_b[j]
                v = 0.999 * v + (1 - 0.999) * (gradient_b[j] ** 2)
                m_hat = m / (1 - 0.9)
                v_hat = v / (1 - 0.999)
                self.b[j] = self.b[j] - learning_rate * \
                    m_hat / (np.sqrt(v_hat) + self.EPSILON)
        else:
            raise ValueError("Invalid optimizer selected.")

        return 0, 0, 0, 0, 0, 0

    def test(self, X, y):
        pred_y = self.forward(X)
        print(pred_y)
        loss = self.compute_cost(y)
        acc = (1.0-np.sum(np.abs(y-np.round(pred_y)))/y.shape[1])*100
        print(f'loss={loss:.5f} accuracy={acc:.2f}%')

    @staticmethod
    def show_result(x, y, pred_y):
        plt.subplot(1, 2, 1)
        plt.title('Ground truth', fontsize=18)
        for i in range(x.shape[0]):
            if y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.subplot(1, 2, 2)
        plt.title('Predict result', fontsize=18)
        for i in range(x.shape[0]):
            if pred_y[i] == 0:
                plt.plot(x[i][0], x[i][1], 'ro')
            else:
                plt.plot(x[i][0], x[i][1], 'bo')
        plt.show()


if __name__ == '__main__':
    X, y = generalize_data.linear(70)
    X = X.T
    y = y.T
    for i in range(3, 4):
        Net_linear = net((np.power(2, i), np.power(2, i)), num_step=1000)
        print('training:')
        time_step, loss_value = Net_linear.train(X, y)
        print('finished\n')
        plt.plot(time_step, loss_value, label=np.power(2, i))
    print('testing:')
    Net_linear.test(X, y)
    print('finished')
    plt.legend(loc='best', title='number of units')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.grid(True)
    plt.show()
    pred_result = Net_linear.forward(X)
    Net_linear.show_result(X.T, y.T, np.round(pred_result).T)

    X, y = generalize_data.xor()
    X = X.T
    y = y.T
    for j in range(3, 4):
        Net_xor = net((np.power(2, j), np.power(2, j)), num_step=1000)
        print('training:')
        time_step, loss_value = Net_xor.train(X, y)
        print('finished\n')
        plt.plot(time_step, loss_value, label=np.power(2, j))
    print('testing:')
    Net_xor.test(X, y)
    print('finished')
    plt.legend(loc='best', title='number of units')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.grid(True)
    plt.show()
    pred_result = Net_xor.forward(X)
    Net_xor.show_result(X.T, y.T, np.round(pred_result).T)
