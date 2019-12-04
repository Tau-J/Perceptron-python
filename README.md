# Perceptron
An implementation of Margin Perceptron, Polynomial Kernel and Gaussian Kernel with pure python codes.

This is a project of CUHK CMSC 5724

> Contributors: 
>
> ​	ZHOU, Shuang 
>
> ​	JIANG, Tao  
>
> ​	DONG, Zichao
>
> ​	LI, Zenan
>
> ​	ZHUANG, Zhende
>
> ​	CUI, Mingyu
##Menu

1. Project Files Preview：
2. Program Usage Guide：
  ----2.1 Required Environment
  ----2.2 Data Generation 
  ----2.3 Run main.py  
3. Function Supplementary Instruction：
  ----3.1 Margin Perceptron
  ----3.2 Polynomial Kernel
  ----3.3 Gaussian Kernel
  ----3.4 Evaluation

##Project File Preview：

├── create_data.py                      // Code for training data and test data generation
├── README.txt                         // Readme file for this project
├── Evaluation.py                        // Code for cross validation, setting criterion and plotting result
├── main.py                                // Main function for whole project
├── margin_perceptron.py           // Code for linear margin perceptron
├── gaussian_kernel.py                // Code for gaussian kernel margin perceptron
├── polynomial_kernel.py            // Code for polynimial kernel margin perceptron
├── new_train_d7.txt                   // Generation training dataset with n=60 and d=3（For alternative choose）
├── new_test_d7.txt                    // Generation test dataset with n=20 and d=3 （For alternative choose）
├── new_train_d9.txt                   // Generation training dataset with n=20000 and d=3
├── new_test_d9.txt                    // Generation test dataset with n=10000 and d=3
├── Contribution Declaration      // Contribution Declaration for each group members

##Program Usage Guide:

#### Required Enviroment:

Python 3.6 or above

```python
import random
import math
import os
```

#### Data Generation:

the steps of generate linear seperable dataset of any dimensions

##### step1: generate a linear seperation plane w

each dimension component in w will randomly generated in range of integer (-99,99). We have fix the random seed so that once the number of dimensions is setted, w is fixed. If you want to generate a dataset according to your designed seperation plane, you need to comment out the code of randomly generating w.

```python
for i in range(dimension):
         w = random.randint(-99, 99)
         w_list.append(w)
         print(f"linear plane: {w_list}")
```

And then use your own w to replace it.

#####step2: generate random points and save them to the point list

Notice that in this part the dimensions values of each point are range in integer (-9999,9999) (you can modify it by changing the code). Also we have set the restriction that there is no repeaded points in our datasets.