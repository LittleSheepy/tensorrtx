#pragma once

#include <dirent.h>
#include <fstream>
#include <unordered_map>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>

// Function to trim leading and trailing whitespace from a string
static inline std::string trim_leading_whitespace(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first) {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}

// Src: https://stackoverflow.com/questions/16605967
static inline std::string to_string_with_precision(const float a_value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

static inline int read_labels(const std::string labels_filename, std::unordered_map<int, std::string>& labels_map) {

    std::ifstream file(labels_filename);
    // Read each line of the file
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        // Strip the line of any leading or trailing whitespace
        line = trim_leading_whitespace(line);

        // Add the stripped line to the labels_map, using the loop index as the key
        labels_map[index] = line;
        index++;
    }
    // Close the file
    file.close();

    return 0;
}

