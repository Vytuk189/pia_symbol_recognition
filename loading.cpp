#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>

#include <fstream>
#include <cstdint>


// Function to manually swap the bytes of a 32-bit integer
uint32_t swap_endian32(uint32_t value) {
    return ((value >> 24) & 0x000000FF) | 
           ((value >> 8)  & 0x0000FF00) | 
           ((value << 8)  & 0x00FF0000) | 
           ((value << 24) & 0xFF000000);
}

// Function to manually swap the bytes of a 16-bit integer
uint16_t swap_endian16(uint16_t value) {
    return ((value >> 8) & 0x00FF) | 
           ((value << 8) & 0xFF00);
}

// Function to read a single 4-byte integer from the file and swap endianness
uint32_t read_uint32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), 4);
    return swap_endian32(value); // Swap from big-endian to little-endian
}

// Function to read a single 2-byte integer from the file and swap endianness
uint16_t read_uint16(std::ifstream& file) {
    uint16_t value;
    file.read(reinterpret_cast<char*>(&value), 2);
    return swap_endian16(value); // Swap from big-endian to little-endian
}

// Structure to store MNIST data
struct MNISTData {
    std::vector<std::vector<uint8_t>> train_images;
    std::vector<uint8_t> train_labels;
    std::vector<std::vector<uint8_t>> test_images;
    std::vector<uint8_t> test_labels;
};


// Function to load MNIST dataset (images and labels) - Training Data
void load_training_data(const std::string& images_path, const std::string& labels_path, MNISTData& data) {
    // Open the images file
    std::ifstream image_file(images_path, std::ios::binary);

    // Open the labels file
    std::ifstream label_file(labels_path, std::ios::binary);

    // Read the header for images
    uint32_t magic_images = read_uint32(image_file);
    uint32_t num_images = read_uint32(image_file);
    uint32_t rows = read_uint32(image_file);
    uint32_t cols = read_uint32(image_file);
    
    // Read the header for labels
    uint32_t magic_labels = read_uint32(label_file);
    uint32_t num_labels = read_uint32(label_file);

    if (num_images != num_labels) {
        std::cerr << "Number of images and labels do not match" << std::endl;
        return;
    }

    // Load the images
    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<uint8_t> image(rows * cols);
        image_file.read(reinterpret_cast<char*>(image.data()), rows * cols);
        data.train_images.push_back(std::move(image));
    }

    // Load the labels
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t label;
        label_file.read(reinterpret_cast<char*>(&label), 1);
        data.train_labels.push_back(label);
    }
}

// Function to load MNIST dataset (images and labels) - Testing Data
void load_testing_data(const std::string& images_path, const std::string& labels_path, MNISTData& data) {
    // Open the images file
    std::ifstream image_file(images_path, std::ios::binary);

    // Open the labels file
    std::ifstream label_file(labels_path, std::ios::binary);

    // Read the header for images
    uint32_t magic_images = read_uint32(image_file);
    uint32_t num_images = read_uint32(image_file);
    uint32_t rows = read_uint32(image_file);
    uint32_t cols = read_uint32(image_file);
    
    // Read the header for labels
    uint32_t magic_labels = read_uint32(label_file);
    uint32_t num_labels = read_uint32(label_file);

    if (num_images != num_labels) {
        std::cerr << "Number of images and labels do not match" << std::endl;
        return;
    }

    // Load the images
    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<uint8_t> image(rows * cols);
        image_file.read(reinterpret_cast<char*>(image.data()), rows * cols);
        data.test_images.push_back(std::move(image));
    }

    // Load the labels
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint8_t label;
        label_file.read(reinterpret_cast<char*>(&label), 1);
        data.test_labels.push_back(label);
    }
}

struct MNISTDataReady {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data;
    std::vector<std::pair<std::vector<double>, uint8_t>> validation_data;
    std::vector<std::pair<std::vector<double>, uint8_t>> test_data;
};

MNISTDataReady convert_to_ready_format(const MNISTData& data) {
    MNISTDataReady ready_data;

    // Convert training data
    for (size_t i = 0; i < 50000; ++i) {
        // Flatten image (28x28) into a 784-dimensional vector
        std::vector<double> flattened_image(784);  // Using double precision
        for (size_t j = 0; j < 784; ++j) {
            flattened_image[j] = static_cast<double>(data.train_images[i][j])/255; 
        }

        std::vector<double> one_hot_label(10, 0);
        one_hot_label[data.train_labels[i]] = 1;

        ready_data.training_data.push_back(std::make_pair(flattened_image, one_hot_label));
    }

    // Convert validation data
    for (size_t i = 50000; i < 60000; ++i) {
        // Flatten image (28x28) into a 784-dimensional vector
        std::vector<double> flattened_image(784);  // Using double precision
        for (size_t j = 0; j < 784; ++j) {
            flattened_image[j] = static_cast<double>(data.train_images[i][j]/255);
        }

        uint8_t label = data.train_labels[i];
        ready_data.validation_data.push_back(std::make_pair(flattened_image, label));
    }

    // Convert test data    
    for (size_t i = 0; i < 10000; ++i) {
        // Flatten image (28x28) into a 784-dimensional vector
        std::vector<double> flattened_image(784);  // Using double precision
        for (size_t j = 0; j < 784; ++j) {
            flattened_image[j] = static_cast<double>(data.test_images[i][j]/255); 
        }

        uint8_t label = data.test_labels[i];
        ready_data.test_data.push_back(std::make_pair(flattened_image, label));
    }

    return ready_data;
}