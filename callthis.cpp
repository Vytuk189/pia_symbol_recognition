#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>
#include "training.h"
#include "callthis.h"
#include <fstream>
#include <cstdint>

int main() {
    // Load the pre-trained network
    Network network = Network::LoadNetwork("network.bin");

    // Read the input vector from the file
    std::ifstream inputFile("input_vector.txt");
    std::vector<double> input;
    double value;
    
    if (inputFile.is_open()) {
        while (inputFile >> value) {
            input.push_back(value);
        }
        inputFile.close();
    } else {
        std::cerr << "Error opening input vector file." << std::endl;
        return 1;
    }

    // Get the output (predicted scores) from the network
    std::vector<double> output = network.Output(input);

    // Open the output file to write the predicted scores
    std::ofstream outputFile("output_vector.txt");
    
    if (outputFile.is_open()) {
        // Write each score to the output file line by line
        for (double score : output) {
            outputFile << score << std::endl;
        }
        outputFile.close(); // Close the file after writing
    } else {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }

    return 0;
}
