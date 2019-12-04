from polynomial_kernel import *
from gaussian_kernel import *
from margin_perceptron import *
from evaluation import *

train_n, train_d, test_n, test_d = 0,0,0,0
train_inputs = []
train_targets = []
test_inputs = []
test_targets = []


# Main
print("Please input the path for training data or use default training data please input number: 1")
train_data_path = input()
print("Please input the path for test data or use default test data please input number: 1")
test_data_path = input()

if train_data_path == '1':
    train_data_path = 'new_train_d9.txt'
if test_data_path == '1':
    test_data_path = 'new_test_d9.txt'

with open(train_data_path, 'r') as fin:
    train_n, train_d = map(int, fin.readline().strip().split())
    for each in fin.readlines():
        each = list(map(int, each.strip().split()))
        train_inputs.append(each[:-1])
        train_targets.append(each[-1])
with open(test_data_path, 'r') as fin:
    test_n, test_d = map(int, fin.readline().strip().split())
    for each in fin.readlines():
        each = list(map(int, each.strip().split()))
        test_inputs.append(each[:-1])
        test_targets.append(each[-1])

print("Please input the running method type: input '1' for linear, '2' for polynomial kernel, '3' for Gaussian kernel")
method_type = input()

if method_type == '1':
    percpt = MarginPerceptron(train_inputs, train_targets, train_n, train_d)
    percpt.train()
    acc = percpt.acc(test_inputs, test_targets)
    print("The accuracy of linear Margin Perceptron is :")
    print(acc)
elif method_type == '2':
    print("Please input the max iteration number")
    max_iter = int(input())
    print("Please input the parameter c")
    c = int(input())
    print(train_n)
    print(train_d)
    percpt = PolynomialPerceptron(train_inputs, train_targets, train_n, train_d, c, max_iter)
    percpt.train()
    acc = percpt.acc(test_inputs, test_targets)
    print("The accuracy of Polynomial Kernel is :")
    print(acc)
elif method_type == '3':
    print("Please input the sigma value you want to test, 1000 is recommended")
    sigma = float(input())
    percpt = GaussianPerceptron(train_inputs, train_targets, train_n, train_d, sigma)
    percpt.train()
    acc = percpt.acc(test_inputs, test_targets)
    print("The accuracy of Gaussian Kernel is :")
    print(acc)
else:
    print("Incorrect input")

# Result and Evaluation
print('Now you can cross-evaluate the models')
print("Please input the model type you want to evaluate input '1' for linear, '2' for polynomial kernel, '3' for Gaussian kernel")
eval_model_type = input()

#
print("Please set the number of folds you want to set for cross-evaluate: ")
number_folds = int(input())
evaluate_init = Evaluator(num_fold=number_folds,model = percpt)

dsets,gts = evaluate_init.get_subsets(train_inputs,train_targets)

if eval_model_type == '1':
    acc_list = []
    for i in range(number_folds):
        train_data = []
        train_gt = []
        eval_data = dsets[i]
        eval_gt = gts[i]
        for j in range(len(dsets)):
            if j != i:
                train_data += dsets[j]
                train_gt += gts[j]
        percpt = MarginPerceptron(train_data, train_gt, len(train_data), len(train_data[0]))
        percpt.train()
        tmp = percpt.acc(eval_data,eval_gt)
        acc_list.append(tmp)
    print('the result of cross_validation is {}'.format(sum(acc_list)/len(acc_list)))
if eval_model_type == '2':
    print("Please input the max iteration number")
    max_iter = int(input())
    print("Please input the parameter c")
    c = int(input())
    acc_list = []
    for i in range(number_folds):
        train_data = []
        train_gt = []
        eval_data = dsets[i]
        eval_gt = gts[i]
        for j in range(len(dsets)):
            if j != i:
                train_data += dsets[j]
                train_gt += gts[j]
        percpt = PolynomialPerceptron(train_data, train_gt, len(train_data), len(train_data[0]), c, max_iter)
        percpt.train()
        tmp = percpt.acc(eval_data,eval_gt)
        acc_list.append(tmp)
    print('the result of cross_validation is {}'.format(sum(acc_list)/len(acc_list)))
elif eval_model_type == '3':
    print("Please input the sigma value you want to test, 1000 is recommended")
    sigma = float(input())
    acc_list = []
    for i in range(number_folds):
        train_data = []
        train_gt = []
        eval_data = dsets[i]
        eval_gt = gts[i]
        for j in range(len(dsets)):
            if j != i:
                train_data += dsets[j]
                train_gt += gts[j]
        percpt = GaussianPerceptron(train_data, train_gt, len(train_data), len(train_data[0]), sigma)
        percpt.train()
        tmp = percpt.acc(eval_data, eval_gt)
        acc_list.append(tmp)
    print('the result of cross_validation is {}'.format(sum(acc_list) / len(acc_list)))

