import os
import time
import numpy as np
from math import cos, sin, pi
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class Random_Grid_SOM:
    def __init__(self, X, city_sequence,no_iteration, no_neuron = 100, learn_rate = 0.02):
        
        self.city_sequence = city_sequence
        self.X = X
        self.no_iteration = no_iteration
        self.learn_rate = learn_rate
        self.no_neuron = no_neuron
        self.wx = np.random.uniform(0, 1, no_neuron)
        self.wy = np.random.uniform(0, 1, no_neuron)
        
    def calculate_distance(self, input_data):
        distance = np.array([])
        for i in range(len(self.wx)):
            dist_i = np.sqrt((input_data[0] - self.wx[i])**2 + (input_data[1] - self.wy[i])**2)
            distance = np.append(distance, dist_i)

        return distance

    def fit(self):
        sigma_zero  = self.no_neuron/2.0
        lambda_t = self.no_iteration / np.log(sigma_zero)
        learn_rate_initial = self.learn_rate
        
        for iteration in (range(self.no_iteration)):
                                                      
            # Select an input data at random
            i1 = iteration % self.X.shape[0]
            input_data = self.X[i1]
                  
            # Calculate the Eucledean distance of this input data from all neuron
            distance = self.calculate_distance(input_data)
                  
            # Find the best matching unit (BMU) neuron index
            bmu_index = distance.argmin()
            
            # Update sigma for neighbourhood radius
            sigma_t = sigma_zero*np.exp(-(float(iteration)/lambda_t)) 
                  
            # Update the learning rate
            self.learn_rate = learn_rate_initial*np.exp(-(float(iteration)/self.no_iteration))
            
            # Adaptation:
            for i in range(len(self.wx)):
                bmu_dist = abs(i - bmu_index)
                if bmu_dist <= sigma_t:
                    tn = np.exp(-((bmu_dist**2)/(2*(sigma_t**2))))
                    self.wx[i] += tn*self.learn_rate*(input_data[0] - self.wx[i])
                    self.wy[i] += tn*self.learn_rate*(input_data[1] - self.wy[i])

            if iteration%1000 == 0 and iteration != 0:
                self.plot_tsp(self.wx, self.wy, iteration)
                if not os.path.exists('Results'):
                    os.mkdir('Results')
                plt.savefig("Results/Fig_iteration{}".format(iteration), dpi = 400)
                plt.show()
                plt.pause(0.0001)              

        return self.wx, self.wy
    
    def plot_tsp(self, wx, wy, itr):
    
        plt.clf()
        plt.scatter(self.X[:,0], self.X[:,1], marker='s', s = 70, color = 'r',\
                    alpha = 0.6)
        for w, x, y in zip(self.city_sequence, self.X[:,0], self.X[:,1]):
            plt.annotate(w,
                         xy=(x, y), xytext=(17, -16),
                         textcoords='offset points', ha='right', va='bottom',
                         #bbox = dict(boxstyle='round,pad=0.1',fc='white', alpha=0.35)
            )
        plt.plot(wx, wy, '-', color = 'b',  linewidth = 2.0, alpha = 0.75)
        plt.axis('off')
        plt.title("Iteration = {}".format(itr))


if __name__ == "__main__":

    infile = open('ulysses22.tsp', 'r')
    infile.readline()
    infile.readline()
    infile.readline()
    Dimension = infile.readline().strip().split()[1]
    infile.readline()
    infile.readline()
    infile.readline()

    X = np.array([])
    No_of_cities = int(Dimension)
    for i in range(No_of_cities):
        x1, y1 = infile.readline().strip().split()[1:]
        X = np.append(X, [float(x1), float(y1)])
    infile.close()
    X = np.reshape(X, (No_of_cities, 2))
    
    city_sequence = ['City {}'.format(x) for x in range(X.shape[0])]
    sc = MinMaxScaler(feature_range = (0, 1))
    X1 = sc.fit_transform(X)
    plt.ion()
    fig = plt.figure(figsize = (9, 9))
    tsp = Random_Grid_SOM(X1, city_sequence = city_sequence,\
                           no_iteration = 10001, \
                           no_neuron = 4*X.shape[0], learn_rate = 0.05)
    tsp.fit()

