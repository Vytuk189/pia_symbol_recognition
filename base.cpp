#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>  
#include <algorithm>

#include <fstream>
#include <cstdint>

#include "base.h"

void Print_Array(const std::vector<int>& array) {
        for (int i = 0; i < array.size(); i++) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    };





void Print_Array(const std::vector<double>& array) {
        for (int i = 0; i < array.size(); i++) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    };


    
