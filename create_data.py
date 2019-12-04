import os
import random


w_list = []
train_blue_pt = []
train_red_pt = []
test_blue_pt = []
test_red_pt = []

# parameters that can be modified
random.seed(1)
dimension = 3
train_blue_pt_count = 10000
train_red_pt_count = 10000
test_blue_pt_count = 5000
test_red_pt_count = 5000
train_data_path = 'new_train_d9.txt'
test_data_path = 'new_test_d9.txt'

# gen a separation plane
for i in range(dimension):
    w = random.randint(-99, 99)
    w_list.append(w)
print(f"linear plane: {w_list}")


while len(train_blue_pt) < train_blue_pt_count or len(train_red_pt) < train_red_pt_count:
    # gen an n-dimension random point
    rand_pt = []
    for k in range(dimension):
        dim_value = random.randint(-9999, 9999)
        rand_pt.append(dim_value)
    # only save the unique point, and discard the point if it is existing in the point list
    if rand_pt in train_red_pt or rand_pt in train_blue_pt:
        continue
    else:
        judge_val = sum([a * b for a, b in zip(w_list, rand_pt)])
        if judge_val > 0 and len(train_blue_pt) < train_blue_pt_count:
            train_blue_pt.append(rand_pt)
            if len(train_blue_pt) % (train_blue_pt_count/10) == 0:
                print(f'train blue pts generation process: {len(train_blue_pt)/train_blue_pt_count*100} %')
        elif judge_val < 0 and len(train_red_pt) < train_red_pt_count:
            train_red_pt.append(rand_pt)
            if len(train_red_pt) % (train_red_pt_count/10) == 0:
                print(f'train red pts generation process: {len(train_red_pt)/train_red_pt_count*100} %')
        else:
            continue

while len(test_blue_pt) < test_blue_pt_count or len(test_red_pt) < test_red_pt_count:
    # gen an n-dimension random point
    rand_pt = []
    for k in range(dimension):
        dim_value = random.randint(-9999, 9999)
        rand_pt.append(dim_value)
    if rand_pt in test_red_pt or rand_pt in test_blue_pt or rand_pt in train_red_pt or rand_pt in train_blue_pt:
        continue
    else:
        judge_val = sum([a * b for a, b in zip(w_list, rand_pt)])
        if judge_val > 0 and len(test_blue_pt) < test_blue_pt_count:
            test_blue_pt.append(rand_pt)
            if len(test_blue_pt) % (test_blue_pt_count/10) == 0:
                print(f'test blue pts generation process: {len(test_blue_pt)/test_blue_pt_count*100} %')
        elif judge_val < 0 and len(test_red_pt) < test_red_pt_count:
            test_red_pt.append(rand_pt)
            if len(test_red_pt) % (test_red_pt_count/10) == 0:
                print(f'test red pts generation process: {len(test_red_pt)/test_red_pt_count*100} %')
        else:
            continue


total_train_pts_count = train_blue_pt_count + train_red_pt_count
total_test_pts_count = test_blue_pt_count + test_red_pt_count

# save the train point
if os.path.exists(train_data_path):
    os.remove(train_data_path)
with open(train_data_path, 'a+') as f:
    f.write("%d %d" % (total_train_pts_count, dimension))
    for pts in train_blue_pt:
        trans_pts = ' '.join(str(elem) for elem in pts)
        f.write('\n%s %s' % (trans_pts, '1'))
    for pts in train_red_pt:
        trans_pts = ' '.join(str(elem) for elem in pts)
        f.write('\n%s %s' % (trans_pts, '0'))

# save the test point
if os.path.exists(test_data_path):
    os.remove(test_data_path)
with open(test_data_path, 'a+') as f:
    f.write("%d %d" % (total_test_pts_count, dimension))
    for pts in test_blue_pt:
        trans_pts = ' '.join(str(elem) for elem in pts)
        f.write('\n%s %s' % (trans_pts, '1'))
    for pts in test_red_pt:
        trans_pts = ' '.join(str(elem) for elem in pts)
        f.write('\n%s %s' % (trans_pts, '0'))
