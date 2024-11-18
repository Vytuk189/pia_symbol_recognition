#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

class Network {
public:

    // Constructor for the neural network
    // Argument - (A, B, C, D): network has A input nodes
    //                          two hidden layers with B and C nodes
    //                          D output nodes
    Network(const std::vector<int>& layer_sizes) {

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
    void Initial_Print() const {
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

    void Print_Array(const std::vector<double>& array) {
        for (int i = 0; i < array.size(); i++) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    };

    double Sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    // Computes the scalar product of two vectors
    double Dot_product(const std::vector<double>& v1, const std::vector<double>& v2) {
        double scalar = 0.0;
        for (int i = 0; i < v1.size(); ++i) {
            scalar += v1[i] * v2[i];
        }
        return scalar;
    }

    // Calculates the output of the network given input, weights and biases
    std::vector<double> Output(const std::vector<double>& input) {

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

     std::vector<std::vector<std::vector<double>>> GetWeights() {
        return Weights;
     }

     std::vector<std::vector<double>> GetBiases() {
        return Biases;
     }


private:
    int Num_layers;
    std::vector<int> Sizes;
    std::vector<std::vector<double>> Biases;  // 2D matrix - each column is biases in one layer
    std::vector<std::vector<std::vector<double>>> Weights; // 3D matrix - each node in a layer has X weights connecting to the X nodes in the previous layer

    // Generate random values between -1 and 1
    double Random_double() const {
        return (std::rand() / (RAND_MAX + 1.0)) * 2 - 1;
    }
};

int main() {
    // Define the sizes of each layer in the network (input, hidden, output)
    std::vector<int> sizes = {2, 3, 1};  // Example: 2 input neurons, 3 hidden neurons, 1 output neuron

    // Create the neural network
    Network net(sizes);

    net.Initial_Print();


    std::vector<double> input = {1., 1.};

    std::vector<double> output = net.Output(input);

    std::cout << "Output is: " << std::endl;
    net.Print_Array(output);




    return 0;
}
