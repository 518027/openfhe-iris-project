#ifndef IRIS_UTILS_H
#define IRIS_UTILS_H

#include <vector>
#include <string>

struct IrisSample {
    std::vector<double> features;
    int label;
};

// Fungsi untuk membaca CSV ke dalam vektor IrisSample
std::vector<IrisSample> loadIrisDataset(const std::string& filename);

// Fungsi untuk normalisasi min-max ke [0,1]
void normalizeDataset(std::vector<IrisSample>& data);

#endif
