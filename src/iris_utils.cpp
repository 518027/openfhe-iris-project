#include "iris_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>

std::vector<IrisSample> loadIrisDataset(const std::string& filename) {
    std::vector<IrisSample> dataset;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Gagal membuka file: " << filename << std::endl;
        return dataset;
    }

    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) { // lewati baris pertama (header)
            header = false;
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        IrisSample sample;
        int col = 0;

        while (std::getline(ss, token, ',')) {
            if (col < 4) {
                sample.features.push_back(std::stod(token));
            } else {
                if (token.find("setosa") != std::string::npos)
                    sample.label = 0;
                else if (token.find("versicolor") != std::string::npos)
                    sample.label = 1;
                else
                    sample.label = 2;
            }
            col++;
        }
        if (sample.features.size() == 4)
            dataset.push_back(sample);
    }
    file.close();
    return dataset;
}

void normalizeDataset(std::vector<IrisSample>& data) {
    if (data.empty()) return;
    int featureCount = data[0].features.size();

    std::vector<double> minVals(featureCount, std::numeric_limits<double>::max());
    std::vector<double> maxVals(featureCount, std::numeric_limits<double>::lowest());

    // Cari min dan max tiap kolom
    for (auto& s : data) {
        for (int i = 0; i < featureCount; i++) {
            minVals[i] = std::min(minVals[i], s.features[i]);
            maxVals[i] = std::max(maxVals[i], s.features[i]);
        }
    }

    // Normalisasi ke [0,1]
    for (auto& s : data) {
        for (int i = 0; i < featureCount; i++) {
            double range = maxVals[i] - minVals[i];
            s.features[i] = (range == 0) ? 0.0 : (s.features[i] - minVals[i]) / range;
        }
    }
}
