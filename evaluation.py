# add code to cross validation  set criterion  and plot the result
import random

class Criterion:
    def get_train_error_rate(self,pred,gt):
        num_train = len(pred)
        count = 0
        for i in range(num_train):
            if pred[i] == gt[i]:
                count += 1
        return count / num_train

    def get_test_error_rate(self,pred,gt):
        num_train = len(pred)
        count = 0
        for i in range(num_train):
            if pred[i] == gt[i]:
                count += 1
        return count / num_train

class Evaluator:
    def __init__(self,num_fold = 3,model = None):
        self.num_fold = num_fold
        self.model = model

    def get_subsets(self,data,gt):
        each_length = len(gt) // self.num_fold
        ret_data = []
        ret_gt = []
        for i in range(self.num_fold - 1):
            tmp_ret_data = []
            tmp_ret_gt = []
            for j in range(each_length):
                tmp = random.randint(0,len(data) - 1)
                tmp_ret_data.append(data.pop(tmp))
                tmp_ret_gt.append(gt.pop(tmp))
            ret_data.append(tmp_ret_data)
            ret_gt.append(tmp_ret_gt)
        ret_data.append(data)
        ret_gt.append(gt)
        return ret_data,ret_gt

    def cross_validation(self,dsets,gts):
        ret_total = []
        for i in range(self.num_fold):
            eval_data = dsets[i]
            eval_gt = gts[i]
            train_data = []
            train_gt = []
            # get train set in each fold
            for j in range(len(dsets)):
                if j != i:
                    train_data += dsets[j]
                    train_gt += gts[j]
#                 train here
            self.model.train(train_data,train_gt)
            tmp_acc = self.model.predict(eval_data,eval_gt)
            correct_num = tmp_acc
            # store the accuarcy in each fold
            ret_total.append(correct_num)
        return sum(ret_total) / len(ret_total)

if __name__ == '__main__':
    # for test
    test_pred = [1,2,0,3]
    test_gt = [1,2,0,3]
    # criterion = Criterion()
    # print(criterion.get_test_error_rate(test_pred,test_gt))
    # test_data = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    # test_dgt = [1,2,3,4,5]
    # evaluator = Evaluator(3)
    # dd,gg = evaluator.get_subsets(test_data,test_dgt)
    # print(dd)
    # print(gg)
