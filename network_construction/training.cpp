
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>
#include <string>
#include "training.h"

#include <fstream>
#include <cstdint>

/*
    This file contains all the necessary functions for creating a neural net, training it and then saving/loading it
*/



// Computes the scalar product of two vectors
double Dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
        double scalar = 0.0;
        if (v1.size() == v2.size()){
        for (int i = 0; i < v1.size(); ++i) {
            scalar += v1[i] * v2[i];
        }
        }else{
            std::cout << "Cant count scalar" << std::endl;
        }
        return scalar;
    }



    // Constructor for the neural network
    // Argument - (A, B, C, D): network has A input nodes
    //                          two hidden layers with B and C nodes
    //                          D output nodes
    Network::Network(const std::vector<int>& layer_sizes) {

        Num_layers = layer_sizes.size();
        Sizes = layer_sizes; // vector of the no. of elements in each layer

        // Initialize biases and weights disregarding the first input layer
        Biases.resize(Num_layers - 1);
        Weights.resize(Num_layers - 1);

        // Necessary for random number generation
        std::srand(std::time(0));

        // Iterates through network layers
        for (int i = 1; i < Num_layers; ++i) {

            // Initialize biases with random values

            // Resize the biases in the i-th layer to a column vector with the appropriate number of elements
            Biases[i - 1].resize(layer_sizes[i], 1);

            for (int j = 0; j < layer_sizes[i]; ++j) {
                // Assign bias values to each node in the ith layer
                Biases[i - 1][j] = Random_double();
            }

            // Initialize weights with random values
            // Same as biases, one extra loop - each node has k weights connecting it to the k nodes in the previous layer
            Weights[i - 1].resize(layer_sizes[i], std::vector<double>(layer_sizes[i - 1], 0));
            for (int j = 0; j < layer_sizes[i]; ++j) {
                for (int k = 0; k < layer_sizes[i - 1]; ++k) {
                    Weights[i - 1][j][k] = Random_double();
                }
            }
        }
    }

    // Function to print network's biases and weights for debugging
    void Network::Initial_Print() const {
        for (int i = 0; i < Num_layers - 1; ++i) {
            std::cout << "Layer " << i + 1 << " Biases:" << std::endl;
            for (int j = 0; j < Biases[i].size(); ++j) {
                std::cout << Biases[i][j] << " ";
            }
            std::cout << std::endl;

            std::cout << "Layer " << i + 1 << " Weights:" << std::endl;
            for (int j = 0; j < Weights[i].size(); ++j) {
                for (int k = 0; k < Weights[i][j].size(); ++k) {
                    std::cout << Weights[i][j][k] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    //Sigmoid function
    double Network::Sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    // Derivative of the Sigmoid function
    double Network::Sigmoid_prime(double z) {
        double sig = Sigmoid(z);
        return sig * (1 - sig);
    }


    // Calculates the output of the network given input, weights and biases
    std::vector<double> Network::Output(const std::vector<double>& input) {

        std::vector<double> output = input;  // Initialize output vector
        
        // Loop through each layer
        for (int i = 0; i < Weights.size(); ++i) {

            // Calculates dot product of weights and input, then add bias
            std::vector<double> temp(Weights[i].size(), 0.0);
            for (int j = 0; j < Weights[i].size(); ++j) {
                temp[j] = Sigmoid(Dot_product(Weights[i][j], output) + Biases[i][j]);
            }
            
            output = temp; // Updates the input for the next layer
        }

        return output;  // Return the output after all layers

    }


    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> Network::backprop(const std::vector<double>& x, const std::vector<double>& y) {
    //void backprop(const std::vector<double>& x, const std::vector<double>& y) {
    // Initialize nabla_b and nabla_w with zero vectors
    std::vector<std::vector<double>> nabla_b(Num_layers - 1);
    std::vector<std::vector<std::vector<double>>> nabla_w(Num_layers - 1);

    for (int i = 1; i < Num_layers; ++i) {
        nabla_b[i-1].resize(Sizes[i], 0.0);
        nabla_w[i-1].resize(Sizes[i], std::vector<double>(Sizes[i-1], 0.0));
    }

    
    // Feedforward
    std::vector<double> activation = x;
    std::vector<std::vector<double>> activations;
    activations.push_back(x);  // Store the input activations

    std::vector<std::vector<double>> zs;

    // Loop through the layers and calculate activations and z's
    for (int i = 0; i < Num_layers - 1; ++i) {
        std::vector<double> z(Sizes[i + 1], 0.0);
        for (int j = 0; j < Sizes[i + 1]; ++j) {
            z[j] = Dot_product(Weights[i][j], activation) + Biases[i][j];
        }
        zs.push_back(z);

        // Sigmoid activation
        std::vector<double> activation_next(Sizes[i + 1]);
        for (int j = 0; j < Sizes[i + 1]; ++j) {
            activation_next[j] = Sigmoid(z[j]);
        }
        activations.push_back(activation_next);
        activation = activation_next;  // Update activation for the next layer
    }
    

    
    
    // Backward pass
    std::vector<double> delta = Cost_derivative(activations.back(), y);
    for (int i = 0; i < delta.size(); ++i) {
        delta[i] *= Sigmoid_prime(zs.back()[i]);
    }

    
    // Initializes the last layer of nabla_b as the cost difference
    nabla_b[Num_layers - 2] = delta;


    
    // Calculate nabla_w for the last layer
    for (int i = 0; i < Sizes[Num_layers - 2]; ++i) {
        for (int j = 0; j < Sizes[Num_layers - 1]; ++j) {
            nabla_w[Num_layers - 2][j][i] = delta[j] * activations[Num_layers - 2][i]; //TADY DOŠLO KE ZMĚNĚ PŘEDTIM Num_layers - 3
        }
    }

    // Loop through the remaining layers
    for (int l = 2; l < Num_layers; ++l) {

        std::vector<double> z = zs[Num_layers - 1 - l];
        std::vector<double> sp(z.size(), 0.0);

        
        for (int i = 0; i < sp.size(); ++i) {
            sp[i] = Sigmoid_prime(z[i]);
        }
        
        
        std::vector<double> delta_next(Sizes[Num_layers - l]);
        for (int i = 0; i < delta_next.size(); ++i) {
            delta_next[i] = 0.0;
            for (int j = 0; j < Sizes[Num_layers - l + 1]; ++j) {
                delta_next[i] += Weights[Num_layers - l][j][i] * delta[j];
            }
            delta_next[i] *= sp[i];
        }
        
        nabla_b[Num_layers - l - 1] = delta_next;

        
        for (int i = 0; i < Sizes[Num_layers - l - 1]; ++i) {
            for (int j = 0; j < Sizes[Num_layers - l]; ++j) {
                nabla_w[Num_layers - l - 1][j][i] = delta_next[j] * activations[Num_layers - l - 1][i];
            }
        }
        
        
        delta = delta_next;  // Update delta for the next iteration

     
    }
    
    
    
    
    return std::make_pair(nabla_b, nabla_w);
    
    }

    
    void Network::update_mini_batch(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& mini_batch, double eta) {

        

        // Initialize nabla_b and nabla_w as zero matrices of the appropriate dimensions
        std::vector<std::vector<double>> nabla_b(Num_layers - 1);
        std::vector<std::vector<std::vector<double>>> nabla_w(Num_layers - 1);
        
        for (int i = 0; i < Num_layers - 1; ++i) {
            nabla_b[i].resize(Sizes[i + 1], 0.0);
            nabla_w[i].resize(Sizes[i + 1], std::vector<double>(Sizes[i], 0.0));
        }

        // Loop over the mini-batch and do backpropagation
        for (const auto& [x, y] : mini_batch) {
            auto [delta_nabla_b, delta_nabla_w] = backprop(x, y);
            
            // Sum the gradients (nabla_b, nabla_w) over the mini-batch
            for (int i = 0; i < Num_layers - 1; ++i) {
                for (int j = 0; j < nabla_b[i].size(); ++j) {
                    nabla_b[i][j] += delta_nabla_b[i][j];
                }
                for (int j = 0; j < nabla_w[i].size(); ++j) {
                    for (int k = 0; k < nabla_w[i][j].size(); ++k) {
                        nabla_w[i][j][k] += delta_nabla_w[i][j][k];
                    }
                }
            }
        }

        // Update the weights and biases using gradient descent
        for (int i = 0; i < Num_layers - 1; ++i) {
            // Update weights
            for (int j = 0; j < Sizes[i + 1]; ++j) {
                for (int k = 0; k < Sizes[i]; ++k) {
                    Weights[i][j][k] -= (eta / mini_batch.size()) * nabla_w[i][j][k];
                }
            }

            // Update biases
            for (int j = 0; j < Sizes[i + 1]; ++j) {
                Biases[i][j] -= (eta / mini_batch.size()) * nabla_b[i][j];
            }
        }
    }

    
    // Function to train the network using Stochastic Gradient Descent
    void Network::SGD(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_data,
             int epochs, int mini_batch_size, double eta,
             std::vector<std::pair<std::vector<double>, uint8_t>>* test_data) {

        int n = training_data.size();
        int n_test = test_data[0].size();

        for (int j = 0; j < epochs; ++j) {
            // Shuffle the training data
            std::shuffle(training_data.begin(), training_data.end(), std::default_random_engine(std::time(0)));

            // Split the training data into mini-batches
            std::vector<std::vector<std::pair<std::vector<double>, std::vector<double>>>> mini_batches;
            for (int k = 0; k < n; k += mini_batch_size) {
                std::vector<std::pair<std::vector<double>, std::vector<double>>> mini_batch;
                for (int i = k; i < std::min(k + mini_batch_size, n); ++i) {
                    mini_batch.push_back(training_data[i]);
                }
                mini_batches.push_back(mini_batch);
            }

            int k = 0;
            // Update the weights and biases for each mini-batch
            std::cout << "Amount of minibatches: " << mini_batches.size() << std::endl;
            for (auto& mini_batch : mini_batches) {
                std::cout << "Updating mini batch " << k << std::endl;
                update_mini_batch(mini_batch, eta);
                if(k==100){
                    break;
                }
                k++;
            }

            
            
            // If test data is provided, evaluate the network's performance
            if (test_data) {
                std::cout << "Epoch " << j << ": " << evaluate(*test_data) << " / " << n_test << std::endl;
            } else {
                std::cout << "Epoch " << j << " complete" << std::endl;
            }
        }
    }

    // Function to find the index of the maximum element
    int Network::findMaxIndex(const std::vector<double>& vec) {
        auto max_it = std::max_element(vec.begin(), vec.end());
        return max_it - vec.begin();  // Return the index of the maximum element
    }
    

    int Network::evaluate(const std::vector<std::pair<std::vector<double>, uint8_t>>& test_data) {
        int correct = 0;
        for (const auto& [x, y] : test_data) {
            std::vector<double> output = Output(x);
            // Evaluate whether the predicted output matches the expected output y
            if (findMaxIndex(output) == y) { // Assuming a simple comparison; in reality, it might be a comparison of max values
                ++correct;
            }
        }
        return correct;
    }

    // Cost function derivative (difference between output activations and target)
    std::vector<double> Network::Cost_derivative(const std::vector<double>& output_activations, const std::vector<double>& y) {
        std::vector<double> diff(output_activations.size());
        for (int i = 0; i < output_activations.size(); ++i) {
            diff[i] = output_activations[i] - y[i];
        }
        return diff;
    }


    // Saves the network into a .txt file that can be read by the user
    void Network::SaveNetwork(const std::string& filename) {

        std::ofstream file(filename, std::ios::trunc);  // Open in truncate mode
        if (!file) {
            std::cerr << "Failed to open file for saving the network." << std::endl;
            return;
        }

        // Save the number of layers
        file << "Num_layers: " << Num_layers << std::endl;

        // Save the sizes of each layer
        file << "Layer_sizes: ";
        for (int i = 0; i < Num_layers; ++i) {
            file << Sizes[i] << " ";
        }
        file << std::endl;

        // Save biases
        for (int i = 0; i < Num_layers - 1; ++i) {
            file << "Layer " << i + 1 << " Biases: ";
            for (int j = 0; j < Sizes[i + 1]; ++j) {
                file << Biases[i][j] << " ";
            }
            file << std::endl;
        }

        // Save weights
        for (int i = 0; i < Num_layers - 1; ++i) {
            file << "Layer " << i + 1 << " Weights:" << std::endl;
            for (int j = 0; j < Sizes[i + 1]; ++j) {
                for (int k = 0; k < Sizes[i]; ++k) {
                    file << Weights[i][j][k] << " ";
                }
                file << std::endl;
            }
        }

        file.close();
        std::cout << "Network saved to " << filename << std::endl;
    }

    // Saves the network into a binary file that will be read for future digit recognitions
    void Network::SaveNetworkBinary(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);  // Open in binary mode
        if (!file) {
            std::cerr << "Failed to open file for saving the network." << std::endl;
            return;
        }

        // Save the number of layers
        file.write(reinterpret_cast<const char*>(&Num_layers), sizeof(Num_layers));

        // Save the sizes of each layer
        for (int size : Sizes) {
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        }

        // Save biases
        for (int i = 0; i < Num_layers - 1; ++i) {
            for (double bias : Biases[i]) {
                file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
            }
        }

        // Save weights
        for (int i = 0; i < Num_layers - 1; ++i) {
            for (int j = 0; j < Sizes[i + 1]; ++j) {
                for (double weight : Weights[i][j]) {
                    file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
                }
            }
        }

        file.close();
        std::cout << "Network saved to " << filename << std::endl;
}

    // Generate random values between -1 and 1
    double Network::Random_double() const {
        return (std::rand() / (RAND_MAX + 1.0)) * 2 - 1;
    }

// Loads the network in binary form to be used for digit evaluation
Network Network::LoadNetwork(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);  // Open in binary mode
    if (!file) {
        std::cerr << "Failed to open file for loading the network." << std::endl;
        throw std::runtime_error("File could not be opened");
    }

    int num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    std::vector<int> layer_sizes(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        file.read(reinterpret_cast<char*>(&layer_sizes[i]), sizeof(layer_sizes[i]));
    }

    Network network(layer_sizes);  // Create the network with the loaded layer sizes

    // Load biases
    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            file.read(reinterpret_cast<char*>(&network.Biases[i][j]), sizeof(network.Biases[i][j]));
        }
    }

    // Load weights
    for (int i = 0; i < num_layers - 1; ++i) {
        for (int j = 0; j < layer_sizes[i + 1]; ++j) {
            for (int k = 0; k < layer_sizes[i]; ++k) {
                file.read(reinterpret_cast<char*>(&network.Weights[i][j][k]), sizeof(network.Weights[i][j][k]));
            }
        }
    }

    file.close();
    //std::cout << "Network loaded from " << filename << std::endl;
    return network;  // Return the loaded network
}



