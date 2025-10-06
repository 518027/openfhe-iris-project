// fhe_svm_iris_multiclass.cpp
// One-vs-rest multiclass demo using OpenFHE CKKS
//
// Compile (WSL, after OpenFHE installed):
// g++ fhe_svm_iris_multiclass.cpp -I/usr/local/include/openfhe -I/usr/local/include/openfhe/core -I/usr/local/include/openfhe/pke -I/usr/local/include/openfhe/binfhe -L/usr/local/lib -lOPENFHEcore -lOPENFHEpke -lOPENFHEbinfhe -std=c++17 -O2 -o fhe_svm_iris_multiclass
//
// Run:
// ./fhe_svm_iris_multiclass
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
        std::cerr << "Gagal membuka file: " << filename << std::endl;
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

// Train one-vs-rest linear models (OLS)
// returns vector of weight vectors (K vectors of size d+1)
std::vector<std::vector<double>> trainOneVsRest(const std::vector<IrisSample>& data, int K) {
    int n = (int)data.size();
    if (n == 0) return {};
    int d = (int)data[0].features.size(); // 4
    int D = d + 1; // include bias

    std::vector<std::vector<double>> allWeights(K, std::vector<double>(D, 0.0));

    for (int cls = 0; cls < K; ++cls) {
        std::vector<std::vector<double>> XtX(D, std::vector<double>(D, 0.0));
        std::vector<double> Xty(D, 0.0);
        for (const auto& s : data) {
            std::vector<double> x(D);
            for (int i = 0; i < d; ++i) x[i] = s.features[i];
            x[d] = 1.0;
            double y = (s.label == cls) ? 1.0 : 0.0;
            for (int i = 0; i < D; ++i) {
                Xty[i] += x[i] * y;
                for (int j = 0; j < D; ++j) XtX[i][j] += x[i] * x[j];
            }
        }
        auto w = solveLinearSystem(XtX, Xty);
        // if solve failed (NaN), fallback to zeros
        allWeights[cls] = w;
    }
    return allWeights;
}

// Encrypted inference: use encrypted weights (one ciphertext per class)
std::pair<int, std::vector<double>> encryptedInferenceMulticlassCKKS(
    CryptoContext<DCRTPoly> cc,
    const KeyPair<DCRTPoly>& keys,
    const std::vector<Ciphertext<DCRTPoly>>& ctWeights, // one ciphertext per class
    const std::vector<double>& features, // d features (no bias)
    uint32_t batchSize
) {
    int d = (int)features.size();
    int K = (int)ctWeights.size();

    // prepare feature vector with bias
    std::vector<double> xvec(d+1);
    for (int i = 0; i < d; ++i) xvec[i] = features[i];
    xvec[d] = 1.0;

    Plaintext px = cc->MakeCKKSPackedPlaintext(xvec);
    auto ctX = cc->Encrypt(keys.publicKey, px);

    std::vector<double> scores(K, 0.0);
    for (int cls = 0; cls < K; ++cls) {
        auto ctRes = cc->EvalInnerProduct(ctWeights[cls], ctX, batchSize);
        Plaintext pres;
        cc->Decrypt(keys.secretKey, ctRes, &pres);
        pres->SetLength(1);
        double score = pres->GetCKKSPackedValue()[0].real();
        scores[cls] = score;
    }

    // choose argmax
    int pred = 0;
    double best = scores[0];
    for (int i = 1; i < K; ++i) {
        if (scores[i] > best) { best = scores[i]; pred = i; }
    }
    return {pred, scores};
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

    // Train one-vs-rest (K=3)
    int K = 3;
    auto allWeights = trainOneVsRest(dataset, K);
    for (int c = 0; c < K; ++c) {
        std::cout << "Weights for class " << c << ": ";
        for (auto w : allWeights[c]) std::cout << w << " ";
        std::cout << std::endl;
    }

    // Setup CKKS (reuse parameters)
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

    KeyPair<DCRTPoly> keys = cc->KeyGen();
    if (!keys.good()) {
        std::cerr << "Key generation gagal." << std::endl;
        return 1;
    }
    cc->EvalMultKeyGen(keys.secretKey);
    cc->EvalSumKeyGen(keys.secretKey);

    // Encrypt each weight vector once (one ciphertext per class)
    std::vector<Ciphertext<DCRTPoly>> ctWeights;
    for (int c = 0; c < K; ++c) {
        Plaintext pw = cc->MakeCKKSPackedPlaintext(allWeights[c]);
        auto ctW = cc->Encrypt(keys.publicKey, pw);
        ctWeights.push_back(ctW);
    }

    // Encrypted inference: test first N samples
    int totalToTest = std::min<int>(150, (int)dataset.size());
    int correct = 0;
    std::cout << std::endl << "Running multiclass encrypted inference on first " << totalToTest << " samples..." << std::endl;

    for (int i = 0; i < totalToTest; ++i) {
        auto pr = encryptedInferenceMulticlassCKKS(cc, keys, ctWeights, dataset[i].features, batchSize);
        int pred = pr.first;
        auto scores = pr.second;
        int trueLabel = dataset[i].label;

        if (pred == trueLabel) ++correct;

        // show first few outputs for inspection
        if (i < 20) {
            std::cout << "Sample " << i << " true=" << trueLabel << " pred=" << pred << " scores=[";
            for (size_t s = 0; s < scores.size(); ++s) {
                std::cout << scores[s] << (s+1<scores.size()? ", " : "");
            }
            std::cout << "]" << std::endl;
        }
    }

    double acc = 100.0 * correct / totalToTest;
    std::cout << "Multiclass encrypted inference accuracy on first " << totalToTest << " samples: " << acc << "%\n";

    // Save results to CSV
    std::ofstream out("results.csv");
    out << "true_label,pred_label,score0,score1,score2\n";
    for (int i = 0; i < totalToTest; ++i) {
        auto pr = encryptedInferenceMulticlassCKKS(cc, keys, ctWeights, dataset[i].features, batchSize);
        int pred = pr.first;
        auto scores = pr.second;
        int trueLabel = dataset[i].label;
        out << trueLabel << "," << pred << "," << scores[0] << "," << scores[1] << "," << scores[2] << "\n";
    }
    out.close();
    std::cout << "Results are saved to results.csv" << std::endl;

    return 0;
}
