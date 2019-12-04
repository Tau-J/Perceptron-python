
class PolynomialPerceptron():
    def __init__(self, inputs, targets, n, d, p, max_iter):
        super(PolynomialPerceptron, self).__init__()
        self.inputs = inputs
        self.targets = targets
        self.p = p
        self.n = n
        self.max_iter = max_iter
        self.alpha = [0 for i in range(n)]

    def kernel_polynomial(self, X, Y, p):
        Y = self.transpose(Y)
        res = self.dot(X, Y)
        res = self.plus(res, Y, 1)
        res = self.exp(res, p)
        return res

    def transpose(self, X):
        if type(X[0]) == int:
            res =[]
            for i in X:
                res.append([i])
            return res
        X = zip(*X)
        X = [list(i) for i in X]
        return X

    def single_multiply(self, X, Y):
        res = 0
        for i in range(len(X)):
            res += X[i] * Y[i]
        return res

    # matrix multiply
    def dot(self, a, b):
        if len(a[0]) == len(b):
            res = [[0] * len(b[0]) for i in range(len(a))]
            for i in range(len(a)):
                for j in range(len(b[0])):
                    for k in range(len(b)):
                        res[i][j] += a[i][k] * b[k][j]
            return res

    # dot like ppt
    def dot_ppt(self, a, b):
        assert len(a) == len(b), 'len(%d) != len(%d)' % (len(a), len(b))
        res = 0
        for idx, each in enumerate(a):
            res += each * b[idx]
        return res

    def plus(self, a, b, flag):
        if flag == 1:
            for i in range(len(a)):
                for m in range(len(a[0])):
                    a[i][m] += 1
        else:
            for i in range(len(a)):
                for m in range(len(a[0])):
                    a[i][m] += b[i][m]
        return a

    def exp(self, a, exp):
        for i in range(exp - 1):
            for m in range(len(a)):
                for n in range(len(a[0])):
                    a[m][n] = a[m][n] * a[m][n]
        return a

    def multiply(self, x, y):
        assert len(x) == len(y), 'len(%d) != len(%d)' % (len(x), len(y))
        for i in range(len(x)):
            x[i] = x[i] * y[i]
        return x

    def train(self):
        K = self.kernel_polynomial(self.inputs, self.inputs, self.p)
        for i in range(self.max_iter):
            for m in range(self.n):
                # a = self.multiply(K[m], self.alpha)
                a = self.multiply(self.alpha, self.targets)
                # tmp = np.dot(a, mp)
                tmp = self.single_multiply(a, K[m])
                if self.targets[m] * tmp <= 0:
                    self.alpha[m] += 1
            print("iteration:",i)


    def predict(self, input_data):
        #   input_data: test data
        # return accuracy of prediction
        # K_val = self.kernel_polynomial(input_data, self.inputs, self.p)
        # res = []
        # for i in range(len(K_val)):
        #     res.append(K_val[i][0])
        tmp = self.single_multiply(self.multiply(self.alpha, self.targets), input_data)
        test_pred = 1 if tmp > 0 else 0
        return test_pred

    def acc(self, inputs, targets):
        #   inputs: test data
        #   targets: test label
        # return accuracy of prediction
        correct = 0
        K_val = self.kernel_polynomial(inputs, self.inputs, self.p)
        for idx, each in enumerate(inputs):
            correct += self.predict(K_val[idx]) == targets[idx]
        return correct / len(inputs)
