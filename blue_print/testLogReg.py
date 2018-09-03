from numpy import *
import matplotlib.pyplot as plt
import time
from LogReg import trainLogRegres, showLogRegres, predicTestData
# from sklearn.datasets import make_circles
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import csv


def loadTrainData():
    train_x = []
    train_y = []
    # fileIn = open('/Users/martinzhang/Desktop/testSet.txt')
    # for line in fileIn.readlines():
    #     lineArr = line.strip().split()
    #     train_x.append([1, float(lineArr[0]), float(lineArr[1])])
    #     train_y.append(float(lineArr[2]))
    # # return mat(train_x), mat(train_y).transpose()
    # for userA, relation in range(relation_dic):
    #     for userB in range(userlist):
    #         if user not in relation:
    #             label = 0
    #         else:
    #             label = 1
    #         train_x.append([1, float(lineArr[0]), float(lineArr[1])])
    #         train_y.append(label)
    # return mat(train_x), mat(train_y).transpose()

    with load('/Users/martinzhang/Desktop/compressed_test_7features.npz') as fd:
        temp_matrix = fd["temptest"]
        # d = [0, 1, 2, 3, 4, 5, 6]
        length =10336114
        a = ones(length)

        temp_matrix_x = temp_matrix[0:length, 0:7]
        result = insert(temp_matrix_x, 0, values=a, axis=1)
        temp_matrix_y = temp_matrix[0:length, 7]
        return mat(result), mat(temp_matrix_y).transpose()







def loadTestData():
    # test_list = []
    # test_id = []
    test_x = []
    # test_y = []
    # file_path = open('/Users/martinzhang/Desktop/ml_data/test-public.txt')



    index = -1
    for line in file_path.readlines():
        index += 1
        if index == 0:
            pass
        else:
            line_content = line.strip().split()
            data_id = line_content[0]
            data_source = line_content[1]
            data_sink = line_content[2]
            # test_list.append([data_id, data_source, data_sink])
            # test_id.append(data_id)
            test_x.append([data_id, data_source, data_sink])
            # test_x
        if index == 100:
            return mat(test_x)

#
#
## step 1: load data
print("step 1: load data...")
train_x, train_y = loadTrainData()
# for item in train_x:
#     print item
# test_list = loadTestData()
# for test_data in test_list:
#     test_x = train_x
# test_y = train_y
# test_x = loadTestData()




## step 2: training...
print("step 2: training...")
opts = {'alpha': 0.01, 'maxIter': 1, 'optimizeType': 'gradDescent'}
optimalWeights = trainLogRegres(train_x, train_y, opts)
# print(optimalWeights.transpose().tolist()[0])
# for line in optimalWeights:
    # print(line[0])
print(optimalWeights.transpose().tolist()[0][1])
print(shape(optimalWeights))



## step 3: predicting
# print("step 3: predicting...")
# accuracy = predicTestData(optimalWeights, test_x, test_y)

# ## step 4: show the result
# print("step 4: show the result...")
# print('The classify accuracy is: %.3f%%' % (accuracy * 100))
# showLogRegres(optimalWeights, train_x, train_y)

# test = loadTestData()
# for item in test:
#     print(item)

# def testCSV():
#     test_y = []
#     # csvfile = open('/Users/martinzhang/Desktop/ml_data/result.csv', 'wb')
#     with open('/Users/martinzhang/Desktop/ml_data/result.csv', 'w') as csvfile:
#         writer = csv.writer(csvfile)
#         for i in range(0, 100):
#             # predict = float(sigmoid(test_x[i, 1:] * weights)[0, 0])
#             wwwww='ggg'
#             # bytes(wwwww, encoding="utf8")
#             writer.writerow((i, wwwww))
#         csvfile.close()
#         print('finishing!')
#
# testCSV()

