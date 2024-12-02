#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>
#include "loading.cpp"
#include "training.cpp"
#include <fstream>
#include <cstdint>

int main() {
    
    // Initialize MNISTData to hold both training and testing data
    MNISTData data;
    
    // Load training data
    load_training_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte", data);
    
    // Load testing data
    load_testing_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", data);
    
    // Print basic information about the dataset
    std::cout << "Training data - Number of images: " << data.train_images.size() << std::endl;
    std::cout << "Training data - Number of labels: " << data.train_labels.size() << std::endl;
    std::cout << "Testing data - Number of images: " << data.test_images.size() << std::endl;
    std::cout << "Testing data - Number of labels: " << data.test_labels.size() << std::endl;
    
    // Convert to the ready format
    MNISTDataReady ready_data = convert_to_ready_format(data);
    
    // Print the first image of the training set (first 28x28 pixel image)
    std::cout << "\nFirst image (28x28 pixel values):" << std::endl;
    const std::vector<double>& image = ready_data.training_data[0].first; // Get the first image
    for (size_t i = 0; i < 28; ++i) {
        for (size_t j = 0; j < 28; ++j) {
            // Print the pixel value, will print 0 for white and 255 for black
            std::cout << (double)image[i * 28 + j] << " "; 
        }
        std::cout << std::endl;
    }
    
    
    std::cout << "Number of pixels in image is: " << image.size() << std::endl;
    // result of the first image
    Print_Array(ready_data.training_data[0].second);



// Define the sizes of each layer in the network (input, hidden, output)
    int n = 784;
    std::vector<int> sizes = {n, 30, 10};  // Example: 2 input neurons, 3 hidden neurons, 1 output neuron

    // Create the neural network
    Network net(sizes);
    /*
    net.Initial_Print();
    std::vector<double> input(n, 1);
    std::vector<double> output = net.Output(input);
    std::cout << "Output: " << std::endl;
    Print_Array(output);
    for(int i = 0; i < 10; ++i){
         std::vector<double> output = net.Output(ready_data.training_data[i].first);
         Print_Array(output);
         net.backprop(ready_data.training_data[i].first, output);
    }
    */

    net.SGD(ready_data.training_data, 1, 10, 2, &ready_data.test_data);
    
    

    std::vector<double> output = net.Output(ready_data.training_data[0].first);
    Print_Array(output);

    net.SaveNetworkBinary("network.bin");    
    net.SaveNetwork("network.txt");

    return 0;
    
}



