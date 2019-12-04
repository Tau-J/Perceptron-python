import math


class GaussianPerceptron():
    def __init__(self, inputs, targets, n, d, sigma):
        super(object, self).__init__()
        assert n == len(inputs), 'number of inputs is not equal to n'
        assert d == len(inputs[0]), 'number of attributes is not equal to d'

        self.w = [0 for i in range(d)]
        self.inputs = inputs
        self.targets = targets
        self.sigma = sigma
        self.final_w = []
        self.final_label = []

    def kernel_gaussian(self, x1, x2, sigma=5.0):
        if self.sigma:
            sigma = self.sigma
        L2_norm = 0
        for d in range(len(x1)):
            L2_norm += (x1[d] - x2[d]) ** 2
        return math.exp(- L2_norm / (2 * (sigma ** 2)))

    def get_label(self, idx):  # map 1/0 to 1/-1
        if self.targets[idx] != int(1):
            label = int(-1)
        else:
            label = self.targets[idx]
        return label

    def train(self):
        global iteration
        iteration = True
        all_w = []
        labels = []
        all_w.append(self.inputs[0])  # the first point is bound to be preserved
        labels.append(self.get_label(0))
        iteration_num = 0
        while iteration:
            for idx, each in enumerate(self.inputs[1:]):
                label = self.get_label(idx+1)
                total_m = 0
                for k in range(len(all_w)):
                    m = self.kernel_gaussian(all_w[k], each)
                    total_m += m * labels[k]   # for violation points, if its label=1, its mapped result will be added
                if total_m * label < 0:
                    all_w.append(self.inputs[idx+1])  # violation, preserve this point
                    labels.append(label)
                    break
                if idx == len(self.inputs)-2:  # so far so good
                    iteration = False
            if iteration_num > 70:  # if iteration over 70, stop it and get result
                iteration = False
            iteration_num += 1
            print('this is a iteration: ', iteration_num)
        print('Finish')
        self.final_w = all_w
        self.final_label = labels

    def predict(self, input_data):
        #   input_data: test data
        # return accuracy of prediction
        total_m = 0
        for k in range(len(self.final_w)):
            m = self.kernel_gaussian(self.final_w[k], input_data)
            total_m += m * self.final_label[k]
        return int(total_m > 0)

    def acc(self, inputs, targets):
        #   inputs: test data
        #   targets: test label
        # return accuracy of prediction
        correct = 0
        for idx, each in enumerate(inputs):
            correct += self.predict(each) == targets[idx]
        return correct / len(inputs)
