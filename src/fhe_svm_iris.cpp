// fhe_svm_iris.cpp
// - reading iris.csv
// - normalizing
// - training simple linear classifier (OLS) for Setosa vs non-Setosa
// - performing encrypted inference using OpenFHE CKKS
//
// Compile (WSL, after openfhe installed):
// g++ fhe_svm_iris.cpp -I/usr/local/include/openfhe -I/usr/local/include/openfhe/core -I/usr/local/include/openfhe/pke -I/usr/local/include/openfhe/binfhe -L/usr/local/lib -lOPENFHEcore -lOPENFHEpke -lOPENFHEbinfhe -std=c++17 -O2 -o fhe_svm_iris
//
// Run:
// ./fhe_svm_iris
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <cmath>

// OpenFHE
#include "pke/openfhe.h"

using namespace lbcrypto;

// Data structures & CSV loader
struct IrisSample {
    std::vector<double> features; // 4 features
    int label; // 0 = setosa, 1 = versicolor, 2 = virginica
};

std::vector<IrisSample> loadIrisDataset(const std::string& filename) {
    std::vector<IrisSample> dataset;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return dataset;
    }

    std::string line;
    bool header = true;
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }
        if (line.size() == 0) continue;

        std::stringstream ss(line);
        std::string token;
        IrisSample sample;
        int col = 0;
        while (std::getline(ss, token, ',')) {
            if (col < 4) {
                try {
                    sample.features.push_back(std::stod(token));
                } catch (...) {
                    sample.features.push_back(0.0);
                }
            } else {
                if (token.find("setosa") != std::string::npos) sample.label = 0;
                else if (token.find("versicolor") != std::string::npos) sample.label = 1;
                else sample.label = 2;
            }
            col++;
        }
        if (sample.features.size() == 4) dataset.push_back(sample);
    }

    file.close();
    return dataset;
}

void normalizeDataset(std::vector<IrisSample>& data) {
    if (data.empty()) return;
    int featureCount = (int)data[0].features.size();
    std::vector<double> minVals(featureCount, std::numeric_limits<double>::max());
    std::vector<double> maxVals(featureCount, std::numeric_limits<double>::lowest());

    for (auto& s : data) {
        for (int i = 0; i < featureCount; ++i) {
            minVals[i] = std::min(minVals[i], s.features[i]);
            maxVals[i] = std::max(maxVals[i], s.features[i]);
        }
    }

    for (auto& s : data) {
        for (int i = 0; i < featureCount; ++i) {
            double range = maxVals[i] - minVals[i];
            s.features[i] = (range == 0.0) ? 0.0 : (s.features[i] - minVals[i]) / range;
        }
    }
}

// Linear algebra helper (Gauss-Jordan)
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b) {
    int n = (int)A.size();
    for (int i = 0; i < n; ++i) A[i].push_back(b[i]); // augment

    for (int i = 0; i < n; ++i) {
        int piv = i;
        for (int r = i; r < n; ++r)
            if (std::fabs(A[r][i]) > std::fabs(A[piv][i])) piv = r;
        std::swap(A[i], A[piv]);
        double diag = A[i][i];
        if (std::fabs(diag) < 1e-12) continue;
        for (int c = i; c <= n; ++c) A[i][c] /= diag;
        for (int r = 0; r < n; ++r) {
            if (r == i) continue;
            double factor = A[r][i];
            for (int c = i; c <= n; ++c) A[r][c] -= factor * A[i][c];
        }
    }

    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = A[i][n];
    return x;
}

// Train linear model (OLS)
// Binary label: setosa -> 1, others -> 0
std::vector<double> trainLinearModel(const std::vector<IrisSample>& data) {
    int n = (int)data.size();
    if (n == 0) return {};
    int d = (int)data[0].features.size(); // 4
    int D = d + 1; // include bias

    std::vector<std::vector<double>> XtX(D, std::vector<double>(D, 0.0));
    std::vector<double> Xty(D, 0.0);

    for (const auto& s : data) {
        std::vector<double> x(D);
        for (int i = 0; i < d; ++i) x[i] = s.features[i];
        x[d] = 1.0; // bias
        double y = (s.label == 0) ? 1.0 : 0.0;

        for (int i = 0; i < D; ++i) {
            Xty[i] += x[i] * y;
            for (int j = 0; j < D; ++j) {
                XtX[i][j] += x[i] * x[j];
            }
        }
    }

    std::vector<double> w = solveLinearSystem(XtX, Xty);
    return w;
}

// Encrypted inference helper
// Encrypt weights once, for each sample encrypt features and EvalInnerProduct
std::pair<int,double> encryptedInferenceCKKS(
    CryptoContext<DCRTPoly> cc,
    const KeyPair<DCRTPoly>& keys,
    const std::vector<double>& weights,
    const std::vector<double>& features,
    uint32_t ringDim, uint32_t dcrtBits, uint32_t multDepth
) {
    int d = (int)features.size();
    uint32_t batchSize = ringDim / 2;

    std::vector<double> wvec(d+1), xvec(d+1);
    for (int i = 0; i < d; ++i) { wvec[i] = weights[i]; xvec[i] = features[i]; }
    wvec[d] = weights[d]; xvec[d] = 1.0; // bias

    // Make plaintexts and encrypt weights once (caller may cache ctWeights if desirable)
    Plaintext pw = cc->MakeCKKSPackedPlaintext(wvec);
    auto ctW = cc->Encrypt(keys.publicKey, pw);

    // encrypt sample features
    Plaintext px = cc->MakeCKKSPackedPlaintext(xvec);
    auto ctX = cc->Encrypt(keys.publicKey, px);

    // inner product
    auto ctRes = cc->EvalInnerProduct(ctW, ctX, batchSize);

    // decrypt
    Plaintext pres;
    cc->Decrypt(keys.secretKey, ctRes, &pres);
    pres->SetLength(1);
    double score = pres->GetCKKSPackedValue()[0].real();

    int pred = (score >= 0.5) ? 1 : 0;
    return {pred, score};
}

// main
int main(int argc, char* argv[]) {
    std::string csvPath = "data/iris.csv";
    auto dataset = loadIrisDataset(csvPath);
    if (dataset.empty()) {
        std::cerr << "The dataset is empty or failed to read. Ensure data/iris.csv exists." << std::endl;
        return 1;
    }
    normalizeDataset(dataset);

    std::cout << "=== Example dataset after normalization ===" << std::endl;
    for (int i = 0; i < 3 && i < (int)dataset.size(); ++i) {
        std::cout << "Sample " << i << ": ";
        for (auto v : dataset[i].features) std::cout << v << " ";
        std::cout << " -> Label: " << dataset[i].label << std::endl;
    }
    std::cout << "==========================================" << std::endl;

    // Train linear classifier (OLS) on whole dataset (for demo)
    auto weights = trainLinearModel(dataset);
    std::cout << "Learned weights (including bias last): ";
    for (auto w : weights) std::cout << w << " ";
    std::cout << std::endl;

    // Setup CKKS parameters
    uint32_t ringDim = 1 << 8;
    uint32_t dcrtBits = 59;
    uint32_t multDepth = 10;
    uint32_t batchSize = ringDim / 2;
    SecurityLevel securityLevel = HEStd_NotSet;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecurityLevel(securityLevel);
    parameters.SetRingDim(ringDim);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    KeyPair keys = cc->KeyGen();
    if (!keys.good()) {
        std::cerr << "Key generation Failed." << std::endl;
        return 1;
    }
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

    // encrypt weights
    std::vector<double> wvec(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) wvec[i] = weights[i];
    Plaintext pw = cc->MakeCKKSPackedPlaintext(wvec);
    auto ctWeights = cc->Encrypt(keys.publicKey, pw);

    // Encrypted inference: test first N samples
    int totalToTest = std::min<int>(50, (int)dataset.size()); // test first 50 samples
    int correct = 0;
    std::cout << std::endl << "Running encrypted inference on first " << totalToTest << " samples..." << std::endl;

    for (int i = 0; i < totalToTest; ++i) {
        // prepare feature vector + bias
        std::vector<double> xvec(dataset[i].features.size() + 1);
        for (size_t j = 0; j < dataset[i].features.size(); ++j) xvec[j] = dataset[i].features[j];
        xvec.back() = 1.0;

        Plaintext px = cc->MakeCKKSPackedPlaintext(xvec);
        auto ctX = cc->Encrypt(keys.publicKey, px);

        // inner product between ctWeights and ctX (both ciphertext) - reuse ctWeights
        auto ctRes = cc->EvalInnerProduct(ctWeights, ctX, batchSize);

        Plaintext pres;
        cc->Decrypt(keys.secretKey, ctRes, &pres);
        pres->SetLength(1);
        double score = pres->GetCKKSPackedValue()[0].real();

        int pred = (score >= 0.5) ? 1 : 0;
        int trueLabel = (dataset[i].label == 0) ? 1 : 0; // setosa -> 1 else 0

        if (pred == trueLabel) ++correct;
        std::cout << "Sample " << i << " true=" << trueLabel << " pred=" << pred << " score=" << score << std::endl;
    }

    double acc = 100.0 * correct / totalToTest;
    std::cout << "Encrypted inference accuracy on first " << totalToTest << " samples: " << acc << "%\n";

    return 0;
}
