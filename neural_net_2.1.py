from numpy import exp, array, random, dot
from math import sin
import random as random_std
import sys

class neuralLayer():
    def __init__(self, aantal_neuronen, aantal_inputs):
        random.seed(1)
        self.weights = 2 * random.random((aantal_neuronen, aantal_inputs))
        

class neuralNet():
    def __init__(self, neural_net_list):
        random_std.seed(1)
        random.seed(1)

        self.layer_list = []
        for x in neural_net_list:
            self.layer_list.append(neuralLayer(x[0], x[1]))

    def train(self, training_iterations, train_input, train_output):
        for x in range(training_iterations):
            if x%(training_iterations/100) == 0:
                print(round(x/training_iterations*100, 2), "%")
            delta_list, change_list = [], []
            output_list = self.think(train_input)
            for x in range(len(self.layer_list) - 1, -1, -1):
                if x == len(self.layer_list) - 1:
                    delta_list.append((train_output - output_list[x]) * self.sigmoid_afgeleide(output_list[x]))
                else:
                    delta_list.append((delta_list[(len(self.layer_list)-2)-x].dot(self.layer_list[x+1].weights.T)) * self.sigmoid_afgeleide(output_list[x]))
            delta_list = delta_list[::-1]
            for x in range(len(self.layer_list)):
                if x == 0:
                    self.layer_list[x].weights += train_input.T.dot(delta_list[x])
                else:
                    self.layer_list[x].weights += output_list[x-1].T.dot(delta_list[x])
            

    def think(self, train_input, print_output = False):
        #print(train_input, self.synaptic_weight_1)
        output_list = []
        for x in range(len(self.layer_list)):
            if x == 0:
                output_list.append(self.sigmoid_function(dot(train_input, self.layer_list[x].weights)))
            else:
                output_list.append(self.sigmoid_function(dot(output_list[x-1], self.layer_list[x].weights)))
        if print_output:
            for x in range(len(ouput_list)):
                print("output layer" + str(x+1) + ":\n", ouput_list[x])
        return(output_list)

    def sigmoid_function(self, x):
        return(1/(1+exp(-x)))

    def sigmoid_afgeleide(self, x):
        return(x*(1-x))
    
    def get_synaptic_weight(self):
        return(self.synaptic_weight_1.T)

    def print_weights(self):
        for x in range(len(self.layer_list)):
            print("layer" + str(x+1) + " weights:\n", self.layer_list[x].weights)

def get_train_set(train_length, train_min, train_max):
    train_input = []
    train_output = []
    for x in range(train_length):
        x = random_std.randint(train_min, train_max)
        train_input.append([x])
        if sin(x) > 0:
            train_output.append([1])
        else:
            train_output.append([0])
    #print(array(train_input), array(train_output))
    return(array(train_input), array(train_output))

if __name__ == "__main__":
    
    train_input, train_output = get_train_set(50, -99, 99)
    train_input = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_output = array([[0, 1, 1, 0]]).T
    test = neuralNet([[3, 3], [3, 5], [5, 3], [3, 1]])
    test.print_weights()
    print("training...")
    #print(train_output)
    test.train(120000, train_input, train_output)
    test.print_weights()
    print("Done!")
    print("[1, 0, 0] geeft 1")
    output = test.think(array([[1, 0, 0]]))
    print(output)
    print("de NNet geeft: ", output[-1])
    print("[0, 0, 0] geeft 0")
    output = test.think(array([[0, 0, 0]]))
    print("de NNet geeft: ", output[-1])
    

