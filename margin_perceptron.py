import math

class MarginPerceptron():
    def __init__(self, inputs, targets, n, d):
        super(MarginPerceptron, self).__init__() 
        assert n == len(inputs), 'number of inputs is not equal to n'
        assert d == len(inputs[0]), 'number of attributes is not equal to d'
        
        self.w = [0 for i in range(d)]
        self.inputs = inputs
        self.targets = targets
        self.R = self.calc_R(inputs)
        self.gamma_guess = self.R
        self.max_epochs = self.calc_max_epochs(self.R, self.gamma_guess)
        
    def calc_R(self, inputs):
        t = []
        for each in inputs:
            t.append(math.sqrt(sum(list(map(lambda x: x*x, each)))))
        return max(t)

    def calc_max_epochs(self, R, gamma_guess):
        return int(12.0*R*R/(gamma_guess*gamma_guess))

    def dot(self, a, b):
        assert len(a) == len(b), 'len(%d) != len(%d)' % (len(a), len(b))
        res = 0
        for idx, each in enumerate(a):
            res += each * b[idx]
        return res

    def train_one_iter(self):
        violation = -1
        for idx, each in enumerate(self.inputs):
            m = self.dot(self.w, each)
            if m * [-1,1][self.targets[idx]] <= 0 or abs(m / math.sqrt(sum(list(map(lambda x: x*x, self.w))))) < self.gamma_guess/2.0 :
                violation = idx
                break
        return violation
    
    def train_once(self):
        for idx in range(self.max_epochs):
            # print('w = %s' % str(self.w))
            violation = self.train_one_iter()
            if violation > -1:
                # target=0 -> w += -1 * p
                # target=1 -> w += 1 * p
                for i in range(len(self.inputs[0])):
                    self.w[i] += [-1,1][self.targets[violation]] * self.inputs[violation][i]
            else:
                return False
        return True

    def train(self):
        while self.train_once():
            self.gamma_guess /= 2.0
            # print(self.gamma_guess)
            self.max_epochs = self.calc_max_epochs(self.R, self.gamma_guess)
            if self.gamma_guess <= 1e-8:
                print('Your data is non-separable')
                return
        print('Finish')


    def predict(self, input_data):
        #   input_data: test data
        # return accuracy of prediction

        return int(self.dot(input_data, self.w) > 0)

    def acc(self, inputs, targets):
        #   inputs: test data
        #   targets: test label
        # return accuracy of prediction

        correct = 0
        for idx, each in enumerate(inputs):
            correct += self.predict(each) == targets[idx]
        return correct / len(inputs)
        