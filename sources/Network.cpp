#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <cblas.h>
#include "../headers/Network.hpp"
#include "../headers/Layer.hpp"

using namespace std;

int Network::getOutputSize(){return Layers[numOfLayers - 1].get_Size();}
int Network::getNumLayers(){return numOfLayers;}
Layer* Network::getLayers(){return Layers;}
double Network::sigmoid(double Z){return 1.0/(1.0 + exp(-1 * Z));}
double Network::sigmoidDrev(double Z){double sigm = sigmoid(Z); return sigm * (1 - sigm);}

void Network::outForward(Layer& curr, Layer& prev)
{
    int M = curr.get_Size();
    int N = prev.get_Size();

    double* W = curr.get_w();
    double* a_prev = prev.get_a();
    double* b = curr.get_b();
    double* a = curr.get_a();

    for (int i = 0; i < M; i++)
        a[i] = b[i];

    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                M, N, 1.0, W, N,
                a_prev, 1, 1.0, a, 1);

    // Use ReLU for hidden layers, sigmoid for output
    bool isOutput = (&curr == &Layers[numOfLayers - 1]);
    for (int i = 0; i < M; i++)
        a[i] = isOutput ? sigmoid(a[i]) : std::max(0.0, a[i]);
}



Network::Network(int numOfLayers, int* netMap) {
    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());

    this->netMap = netMap;
    this->made_here = false;
    this->numOfLayers = numOfLayers;
    this->Layers = new Layer[numOfLayers];

    for (int i = 0; i < numOfLayers; i++) {
        this->Layers[i].set_Size(netMap[i]);
        this->Layers[i].set_a(new double[netMap[i]]);

        if (i > 0) {
            int fan_in = netMap[i - 1];
            int fan_out = netMap[i];
            double limit = sqrt(6.0 / (fan_in + fan_out)); // Xavier range
            std::uniform_real_distribution<double> dist(-limit, limit);

            double* b = new double[fan_out];
            double* w = new double[fan_out * fan_in];

            for (int j = 0; j < fan_out; ++j)
                b[j] = 0.0; // You can also randomize these if needed

            for (int j = 0; j < fan_out * fan_in; ++j)
                w[j] = dist(generator); // Proper initialization

            this->Layers[i].set_b(b);
            this->Layers[i].set_w(w);
        }
    }
}

Network::Network(string FilePath)
{
    std::ifstream file(FilePath, std::ios::binary);
    if (!file.is_open())
    {
        cerr << "Failed to load network from file!" << endl;
        this->numOfLayers = 0;
        this->Layers = nullptr;
        return;
    }

    file.read(reinterpret_cast<char*>(&numOfLayers), sizeof(int));
    this->Layers = new Layer[numOfLayers];

    this->netMap = new int[numOfLayers];
    this->made_here = true;
    file.read(reinterpret_cast<char*>(netMap), sizeof(int) * numOfLayers);

    for (int i = 0; i < numOfLayers; i++)
    {
        this->Layers[i].set_Size(netMap[i]);
        this->Layers[i].set_a(new double[netMap[i]]);
    }

    for (int i = 1; i < numOfLayers; i++)
    {
        double* b = new double[netMap[i]];
        double* w = new double[netMap[i] * netMap[i-1]];

        file.read(reinterpret_cast<char*>(b), sizeof(double) * netMap[i]);
        file.read(reinterpret_cast<char*>(w), sizeof(double) * netMap[i] * netMap[i-1]);

        this->Layers[i].set_b(b);
        this->Layers[i].set_w(w);
    }

    file.close();
}

bool Network::store(string FilePath)
{
    ofstream file(FilePath, ios::binary);
    if(!file.is_open())
    {
        cerr << "Failed to store network!" << endl;
        return false;
    }
    else
    {
        file.write(reinterpret_cast<const char*>(&numOfLayers), sizeof(int));
        file.write(reinterpret_cast<const char*>(this->netMap), sizeof(int) * numOfLayers);

        for(int i=1; i<numOfLayers; i++)
        {
            int biasSize = Layers[i].get_Size();
            double* bias = Layers[i].get_b();
            file.write(reinterpret_cast<const char*>(bias), sizeof(double) * biasSize);

            int weightSize = Layers[i].get_Size() * Layers[i-1].get_Size();
            double* weight = Layers[i].get_w();
            file.write(reinterpret_cast<const char*>(weight), sizeof(double) * weightSize);
        }
    }
    return true;
}

double* softmax_inplace(double* output, int size) {
    // Find the max for numerical stability
    double maxVal = output[0];
    for (int i = 1; i < size; ++i) {
        if (output[i] > maxVal)
            maxVal = output[i];
    }

    // Compute exponentials and sum
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(output[i] - maxVal); // safe from overflow
        sum += output[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; ++i) {
        output[i] /= sum;
    }

    return output;
}


double* Network::output(double* input)
{
    // copy input to the first layer's activation
    double* a0 = Layers[0].get_a();
    int inputSize = Layers[0].get_Size();
    for (int i = 0; i < inputSize; i++)
        a0[i] = input[i];

    for (int i = 1; i < numOfLayers; i++)
        outForward(Layers[i], Layers[i - 1]);
#ifdef SOFTMAX
    return softmax_inplace(Layers[numOfLayers - 1].get_a(), Layers[numOfLayers - 1].get_Size());
#endif // SOFTMAX
    return Layers[numOfLayers - 1].get_a();
}

Network::~Network()
{
    for (int i = 0; i < numOfLayers; ++i)
    {
        double* a = Layers[i].get_a();
        double* b = Layers[i].get_b();
        double* w = Layers[i].get_w();

        if (a) delete[] a;
        if (b) delete[] b;
        if (w) delete[] w;
    }
    if(netMap && made_here) delete[] netMap;
    if(Layers) delete[] Layers;
}

