#ifndef LOADING_H
#define LOADING_H

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>

uint32_t swap_endian32(uint32_t value);
uint16_t swap_endian16(uint16_t value);
uint32_t read_uint32(std::ifstream& file);
uint16_t read_uint16(std::ifstream& file);
struct MNISTData {
    std::vector<std::vector<uint8_t>> train_images;  // 2D vector for training images
    std::vector<uint8_t> train_labels;               // Vector for training labels
    std::vector<std::vector<uint8_t>> test_images;   // 2D vector for test images
    std::vector<uint8_t> test_labels;                // Vector for test labels
};
void load_training_data(const std::string& images_path, const std::string& labels_path, MNISTData& data);
void load_testing_data(const std::string& images_path, const std::string& labels_path, MNISTData& data);
struct MNISTDataReady {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data;
    std::vector<std::pair<std::vector<double>, uint8_t>> validation_data;
    std::vector<std::pair<std::vector<double>, uint8_t>> test_data;
};
MNISTDataReady convert_to_ready_format(const MNISTData& data);


#endif