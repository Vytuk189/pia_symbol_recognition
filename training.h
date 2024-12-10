#ifndef TRAINING_H
#define TRAINING_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>
#include <string>
#include "base.cpp"

#include <fstream>
#include <cstdint>

double Dot_product(const std::vector<double>& v1, const std::vector<double>& v2);

class Network {
public:

    // Constructor for the neural network
    // Argument - (A, B, C, D): network has A input nodes
    //                          two hidden layers with B and C nodes
    //                          D output nodes
    Network(const std::vector<int>& layer_sizes);

    // Function to print network's biases and weights for debugging
    void Initial_Print() const;

    double Sigmoid(double z);

    // Derivative of the Sigmoid function
    double Sigmoid_prime(double z);

    // Calculates the output of the network given input, weights and biases
    std::vector<double> Output(const std::vector<double>& input);

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> backprop(const std::vector<double>& x, const std::vector<double>& y);

    void update_mini_batch(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch, double eta);

    // Function to train the network using Stochastic Gradient Descent
    void SGD(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_data,
             int epochs, int mini_batch_size, double eta,
             std::vector<std::pair<std::vector<double>, uint8_t>>* test_data = nullptr);

    // Function to find the index of the maximum element
    int findMaxIndex(const std::vector<double>& vec);
    
    int evaluate(const std::vector<std::pair<std::vector<double>, uint8_t>>& test_data);

    // Cost function derivative (difference between output activations and target)
    std::vector<double> Cost_derivative(const std::vector<double>& output_activations, const std::vector<double>& y);

    void SaveNetwork(const std::string& filename);

    void SaveNetworkBinary(const std::string& filename);

    static Network LoadNetwork(const std::string& filename);

private:
    int Num_layers;
    std::vector<int> Sizes;
    std::vector<std::vector<double>> Biases;  // 2D matrix - each column is biases in one layer
    std::vector<std::vector<std::vector<double>>> Weights; // 3D matrix - each node in a layer has X weights connecting to the X nodes in the previous layer

    // Generate random values between -1 and 1
    double Random_double() const;

};

#endif