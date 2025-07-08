#include <cblas.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include "NeuralTrainer.hpp"

NeuralTrainer::NeuralTrainer(Network* network)
{
    this->network = network;
    numOfLayers = this->network->getNumLayers();

    this->outErr = new double*[numOfLayers];
    for(unsigned int i=1; i < numOfLayers; i++)
        this->outErr[i] = new double[this->network->getLayers()[i].get_Size()];

    this->reqOut = this->inputs = nullptr;
    this->accuracy = this->loss = nullptr;
    this->miniBatSize = this->eta = this->lambda = this->numOfInputs = 0;
}

void NeuralTrainer::setInputs(double** inputs, unsigned int numOfInputs)
{
    this->inputs = inputs;
    this->numOfInputs = numOfInputs;
}

void NeuralTrainer::outErrLastLayer(int reqOutIndex)
{
    double* crctOut = this->reqOut[reqOutIndex];
    double* out = network->getLayers()[numOfLayers - 1].get_a();
    int len = network->getLayers()[numOfLayers-1].get_Size();

    for(int i=0; i<len; i++)
        outErr[numOfLayers - 1][i] = out[i] - crctOut[i];

}

void NeuralTrainer::outErrLayer(unsigned int LayerNum, int reqOutIndex)
{
    if (LayerNum == this->numOfLayers - 1)
    {
        outErrLastLayer(reqOutIndex);
        return;
    }
    else if (LayerNum == 0)
    {
        return;
    }
    else
    {
        Layer* curr = &network->getLayers()[LayerNum];
        Layer* next = &network->getLayers()[LayerNum + 1];

        int currSize = curr->get_Size();
        int nextSize = next->get_Size();

        double* currA = curr->get_a();
        double* Wnext = next->get_w(); // shape: nextSize x currSize (row major)
        double* deltaNext = outErr[LayerNum + 1];
        double* deltaCurr = outErr[LayerNum];

        // Initialize error to 0
        for (int i = 0; i < currSize; ++i)
            deltaCurr[i] = 0.0;

        // deltaCurr[i] = sum_j (Wnext[j * currSize + i] * deltaNext[j])
        for (int j = 0; j < nextSize; ++j)
        {
            for (int i = 0; i < currSize; ++i)
            {
                deltaCurr[i] += Wnext[j * currSize + i] * deltaNext[j];
            }
        }

        // Hadamard product with sigmoid derivative
        for (int i = 0; i < currSize; ++i)
        {
            // ReLU derivative: f'(x) = 1 if x > 0 else 0
            deltaCurr[i] *= (currA[i] > 0.0) ? 1.0 : 0.0;
        }
    }
}

void NeuralTrainer::updateBiases()
{
    for (unsigned int l = 1; l < numOfLayers; ++l)
    {
        double* b = network->getLayers()[l].get_b();
        double* delta = outErr[l];
        int layerSize = network->getLayers()[l].get_Size();

        for (int i = 0; i < layerSize; ++i)
        {
            b[i] -= (eta / miniBatSize) * delta[i];
        }
    }
}

void NeuralTrainer::updateWeights()
{
    for (unsigned int l = 1; l < numOfLayers; ++l)
    {
        Layer* layer = &network->getLayers()[l];
        Layer* prev = &network->getLayers()[l - 1];

        double* w = layer->get_w();            // Weight matrix (curr × prev)
        double* delta = outErr[l];             // Delta for current layer (curr)
        double* a_prev = prev->get_a();        // Activations from previous layer (prev)

        int currSize = layer->get_Size();
        int prevSize = prev->get_Size();

        // Apply L2 decay: w := w * (1 - eta * lambda / m)
        double decayFactor = 1.0 - (eta * lambda / miniBatSize);
        for (int i = 0; i < currSize * prevSize; ++i)
            w[i] *= decayFactor;

        // Apply gradient update: w := w - eta/m * delta * a_prev^T
        double scale = -eta / miniBatSize;
        cblas_dger(CblasRowMajor,
                   currSize,         // M (rows of W)
                   prevSize,         // N (cols of W)
                   scale,            // alpha = - / m
                   delta, 1,         // delta: Mx1
                   a_prev, 1,        // a_prev: 1xN
                   w, prevSize);     // W matrix in row-major
    }
}

int argmax(double* vec, int size)
{
    int maxIndex = 0;
    for (int i = 1; i < size; ++i)
    {
        if (vec[i] > vec[maxIndex])
            maxIndex = i;
    }
    return maxIndex;
}

void shuffleIndices(std::vector<int>& indices)
{
    static std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(indices.begin(), indices.end(), rng);
}

void NeuralTrainer::train(unsigned int epochs, double eta, double lambda, unsigned int miniBatchSize)
{
    this->eta = eta;
    this->lambda = lambda;
    this->miniBatSize = miniBatchSize;
    this->epochs = epochs;
    this->accuracy = new double[epochs];
    this->loss = new double[epochs];

    const int outputSize = network->getOutputSize();

    // Compute label counts to get class frequencies
    std::vector<int> labelCounts(outputSize, 0);
    for (unsigned int i = 0; i < numOfInputs; ++i)
        for (int k = 0; k < outputSize; ++k)
            if (reqOut[i][k] >= 0.5)
                labelCounts[k]++;

    // Compute inverse frequency weights
    std::vector<double> labelWeights(outputSize);
    for (int k = 0; k < outputSize; ++k)
    {
        double freq = labelCounts[k] / static_cast<double>(numOfInputs);
        labelWeights[k] = freq > 0.0 ? (1.0 / freq) : 1.0;  // avoid div-by-zero
    }

    std::vector<int> indices(numOfInputs);
    for (unsigned int i = 0; i < numOfInputs; ++i)
        indices[i] = i;

    for (unsigned int epoch = 0; epoch < epochs; ++epoch)
    {
        shuffleIndices(indices);

        double totalLoss = 0.0;
        int correctLabelCount = 0;
        std::vector<int> correctPerLabel(outputSize, 0);

        for (unsigned int i = 0; i < numOfInputs; i += miniBatchSize)
        {
            int batchEnd = std::min(i + miniBatchSize, numOfInputs);

            // Reset error buffers
            for (unsigned int l = 1; l < numOfLayers; ++l)
                std::fill(outErr[l], outErr[l] + network->getLayers()[l].get_Size(), 0.0);

            for (int j = i; j < batchEnd; ++j)
            {
                int idx = indices[j];
                double* output = network->output(inputs[idx]);

                for (int k = 0; k < outputSize; ++k)
                {
                    double pred = std::clamp(output[k], 1e-7, 1.0 - 1e-7); // prevent log(0)
                    double target = reqOut[idx][k];
                    double weight = labelWeights[k];

                    totalLoss += weight * (-(target * std::log(pred) + (1 - target) * std::log(1 - pred)));

                    int predBinary = (output[k] >= 0.5);
                    int targetBinary = (target >= 0.5);
                    if (predBinary == targetBinary)
                    {
                        correctLabelCount++;
                        correctPerLabel[k]++;
                    }
                }

                outErrLastLayer(idx);
                for (int l = numOfLayers - 2; l >= 1; --l)
                    outErrLayer(l, idx);
            }

            updateBiases();
            updateWeights();
        }

        double avgLoss = totalLoss / (numOfInputs * outputSize);
        double acc = 100.0 * correctLabelCount / (numOfInputs * outputSize);
        this->loss[epoch] = avgLoss;
        this->accuracy[epoch] = acc;

        cout << "Epoch " << (epoch + 1) << "/" << epochs
             << " - Accuracy: " << acc << "% - Loss: " << avgLoss << endl;

        eta = getDecayedEta(eta, epoch);
    }


    cout << "Training complete.\n";
}


bool NeuralTrainer::storeLoss(string FilePath)
{
    ofstream file(FilePath, ios::binary);
    if(!file.is_open())
    {
        cerr << "Failed to store network!" << endl;
        return false;
    }
    else
    {
        file.write(reinterpret_cast<const char*>(&epochs), sizeof(int));
        file.write(reinterpret_cast<const char*>(this->accuracy), sizeof(double) * epochs);
        file.write(reinterpret_cast<const char*>(this->loss), sizeof(double) * epochs);
    }
    return true;
}

NeuralTrainer::~NeuralTrainer()
{
    if (outErr)
    {
        for (unsigned int i = 1; i < numOfLayers; i++)
        {
            if (outErr[i])
                delete[] outErr[i];
        }
        delete[] outErr;
    }

    if (reqOut)
        delete[] reqOut;
}

