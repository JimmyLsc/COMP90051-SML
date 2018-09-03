from numpy import *
import matplotlib.pyplot as plt
import time
# from sklearn.datasets import make_circles
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
import csv


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def trainLogRegres(train_x, train_y, opts):
    startTime = time.time()
    numSamples, numFeatures = shape(train_x)
    print("numSamples", numSamples)
    print("numFeatures", numFeatures)
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = ones((numFeatures, 1))
    print(shape(weights))
    weights_list = []
    for k in range(maxIter):
        print("Iter", k)
        if opts['optimizeType'] == 'gradDescent':  # gradient descent algorilthm
            output = sigmoid(train_x * weights)
            error = train_y - output
            weights = weights + alpha * train_x.transpose() * error
            print("weights", weights)
            print("weights", weights.transpose().tolist()[0])
            weights_list.append(weights.transpose().tolist()[0])
        elif opts['optimizeType'] == 'stocGradDescent':  # stochastic gradient descent
            for i in range(numSamples):
                if i % 20000 == 0:
                    print("Iter + index: ", k, i)
                output = sigmoid(train_x[i, :] * weights)
                # print('output', output)
                error = train_y[i, 0] - output
                # print('error', error)
                weights = weights + alpha * train_x[i, :].transpose() * error
                weights_list.append(weights.transpose().tolist()[0])
                # print('weights', weights)
        elif opts['optimizeType'] == 'smoothStocGradDescent':  # smooth stochastic gradient
            dataIndex = list(range(numSamples))
            for i in range(numSamples):
                print("iter, index: ", k, i)
                # print(": ", i)
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del dataIndex[randIndex] # during one interation, delete the optimized sample
                weights_list.append(weights.transpose().tolist()[0])
        else:
            raise NameError('Not support optimize method type!')
    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    return weights, weights_list


def predicTestData(weights, test_x):
    test_data_length, number_of_features = shape(test_x)
    # matchCount = 0
    result = []
    for i in range(test_data_length):
        # print('test_x[i, :]', test_x[i, :])
        # print('weights', weights)
        predict = float(sigmoid(test_x[i, :] * weights)[0, 0])
        # result.append(i+1)
        result.append(predict)
        print(i+1, predict)
    with open('final_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Id, Prediction
        line_title = []
        line_title.append('Id')
        line_title.append('Prediction')
        writer.writerow(line_title)
        index = 0
        for line in result:
            index += 1
            temp_list = []
            temp_list.append(index)
            temp_list.append("%.9f" % line)
            writer.writerow(temp_list)
        csvfile.close()
        print('finishing!')

def showWeightDiagram(weight_list):
    x_values = list(range(1, len(weight_list)+1))
    y0_values = [value[0] for value in weight_list]
    y1_values = [value[1] for value in weight_list]
    y2_values = [value[2] for value in weight_list]
    y3_values = [value[3] for value in weight_list]
    y4_values = [value[4] for value in weight_list]
    y5_values = [value[5] for value in weight_list]
    y6_values = [value[6] for value in weight_list]
    y7_values = [value[7] for value in weight_list]
    y8_values = [value[8] for value in weight_list]
    y9_values = [value[9] for value in weight_list]
    y10_values = [value[10] for value in weight_list]
    y11_values = [value[11] for value in weight_list]
    y12_values = [value[12] for value in weight_list]
    plt.plot(x_values, y0_values, label='W0')
    plt.plot(x_values, y1_values, label='W1')
    plt.plot(x_values, y2_values, label='W2')
    plt.plot(x_values, y3_values, label='W3')
    plt.plot(x_values, y4_values, label='W4')
    plt.plot(x_values, y5_values, label='W5')
    plt.plot(x_values, y6_values, label='W6')
    plt.plot(x_values, y7_values, label='W7')
    plt.plot(x_values, y8_values, label='W8')
    plt.plot(x_values, y9_values, label='W9')
    plt.plot(x_values, y10_values, label='W10')
    plt.plot(x_values, y11_values, label='W11')
    plt.plot(x_values, y12_values, label='W12')
    plt.legend()
    plt.show()
    plt.savefig('weights.png', dpi=300)
