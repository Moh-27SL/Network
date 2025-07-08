#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "../headers/Network.hpp"
#include "../headers/NeuralTrainer.hpp"

using namespace std;

bool load_npy_uint8(const string& filename, vector<unsigned char>& data, vector<unsigned long>& shape) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        return false;
    }

    string magic(6, '\0');
    file.read(&magic[0], 6);
    if (magic != "\x93NUMPY") {
        cerr << "Invalid npy file: " << filename << endl;
        return false;
    }

    char major, minor;
    file.read(&major, 1);
    file.read(&minor, 1);

    uint16_t header_len;
    file.read(reinterpret_cast<char*>(&header_len), 2);

    string header(header_len, '\0');
    file.read(&header[0], header_len);

    size_t pos1 = header.find('(');
    size_t pos2 = header.find(')');
    string shapeStr = header.substr(pos1 + 1, pos2 - pos1 - 1);

    shape.clear();
    size_t start = 0;
    while (start < shapeStr.size()) {
        size_t end = shapeStr.find(',', start);
        if (end == string::npos) end = shapeStr.size();
        shape.push_back(stoul(shapeStr.substr(start, end - start)));
        start = end + 1;
    }

    size_t total = 1;
    for (auto s : shape) total *= s;
    data.resize(total);
    file.read(reinterpret_cast<char*>(data.data()), total);
    return true;
}

double** flatten_images(const vector<vector<vector<uint8_t>>>& imgs) {
    int n = imgs.size();
    int h = imgs[0].size();
    int w = imgs[0][0].size();
    double** output = new double*[n];
    for (int i = 0; i < n; ++i) {
        output[i] = new double[h * w];
        int k = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                output[i][k++] = imgs[i][y][x] / 255.0;
    }
    return output;
}

double** load_labels_all(const string& folder, const string& split, int count) {
    double** labels = new double*[count];
    for (int i = 0; i < count; ++i)
        labels[i] = new double[4];

    vector<string> parts = {"acrosome", "head", "vacuole", "tail"};
    for (int j = 0; j < 4; ++j) {
        vector<unsigned char> flat;
        vector<unsigned long> shape;
        string file = folder + "y_" + parts[j] + "_" + split + ".npy";
        if (!load_npy_uint8(file, flat, shape)) {
            cerr << "Failed to load: " << file << endl;
            continue;
        }
        assert(shape[0] == count);
        for (int i = 0; i < count; ++i)
            labels[i][j] = flat[i]; // either 0 or 1
    }
    return labels;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <cstdint>
#include "../headers/Network.hpp"
#include "../headers/NeuralTrainer.hpp"

using namespace std;

// (… your load_npy_uint8, flatten_images, load_labels_all as before …)

int main() {
    string folder = "E:\\mohammed\\neural networks\\sperm_detektion\\mhsma-dataset-master\\mhsma\\";

    // — Load and flatten training images —
    vector<unsigned char> flat_train;
    vector<unsigned long> shape_train;
    load_npy_uint8(folder + "x_64_train.npy", flat_train, shape_train);
    int N = shape_train[0], H = shape_train[1], W = shape_train[2];
    vector<vector<vector<uint8_t>>> images(N, vector<vector<uint8_t>>(H, vector<uint8_t>(W)));
    for (int i = 0; i < N; ++i)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                images[i][y][x] = flat_train[i * H * W + y * W + x];
    double** inputs = flatten_images(images);

    // — Load multilabel targets —
    double** labels = load_labels_all(folder, "train", N);

        // Add this in main() after loading labels:
    for (int k = 0; k < 4; ++k) {
        int count = 0;
        for (int i = 0; i < N; ++i)
            if (labels[i][k] == 1.0)
                count++;
        cout << "Label " << k << ": " << count << "/" << N << " are 1s\n";
    }


    // — Build your network & trainer —
    int netMap[] = {64*64, 30, 4};
    Network net(3, netMap);

    NeuralTrainer trainer(&net);
    trainer.setInputs(inputs, N);
    trainer.SetreqOut(labels);

    // — Train with a smaller eta and smaller batch size —
    trainer.train(
      /*epochs*/       100,
      /*eta*/          3,
      /*lambda*/       0.001,
      /*miniBatchSize*/32
    );

    // — Store trained model & loss curve —
    net.store("sperm_detector_multilabel.nun");
    trainer.storeLoss("sperm_detector_multilabel_graph.bin");

    // — Compute per–label accuracy on TRAINING set —
    int M = net.getOutputSize();        // should be 4
    vector<int> correct(M, 0), total(M, 0);

    for (int i = 0; i < N; ++i) {
        double* out = net.output(inputs[i]);
        for (int k = 0; k < M; ++k) {
            int p = out[k] >= 0.5;
            int t = labels[i][k] >= 0.5;
            if (p == t) correct[k]++;
            total[k]++;
        }
    }

    static const char* names[4] = {"Acrosome", "Head", "Vacuole", "Tail"};
    cout << "\nPer-label accuracy on training set:\n";
    for (int k = 0; k < M; ++k) {
        double acc = 100.0 * correct[k] / total[k];
        cout << "  " << names[k] << ": " << acc << "%\n";
    }

    // — Overall multilabel exact-match accuracy —
    int exactMatch = 0;
    for (int i = 0; i < N; ++i) {
        double* out = net.output(inputs[i]);
        bool allok = true;
        for (int k = 0; k < M; ++k)
            if (((out[k] >= 0.5) != (labels[i][k] >= 0.5))) { allok = false; break; }
        if (allok) exactMatch++;
    }
    cout << "Exact match (all 4 correct): "
         << (100.0 * exactMatch / N) << "%\n";

    return 0;
}
