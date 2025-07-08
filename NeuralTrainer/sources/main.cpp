#include <iostream>
#include <cmath>
#include <vector>
#include "../../headers/Network.hpp"

using namespace std;

bool compareOutputs(double* out1, double* out2, int size, double epsilon = 1e-9)
{
    for (int i = 0; i < size; ++i)
    {
        if (fabs(out1[i] - out2[i]) > epsilon)
            return false;
    }
    return true;
}

int main()
{
    // Big network: 7 layers
    int netMap[] = {16, 64, 128, 64, 16, 8, 1};
    int numLayers = sizeof(netMap) / sizeof(int);

    // Generate random input for the 16 input neurons
    vector<double> input(netMap[0]);
    for (int i = 0; i < netMap[0]; ++i)
        input[i] = (double) rand() / RAND_MAX;

    cout << "Creating BIG network..." << endl;
    Network net(numLayers, netMap);

    double* output1 = net.output(input.data());

    cout << "Big Network Output: ";
    for (int i = 0; i < net.getOutputSize(); ++i)
        cout << output1[i] << " ";
    cout << endl;

    // Save to file
    cout << "Storing network to file..." << endl;
    net.store("big_network.bin");

    cout << "Loading network from file..." << endl;
    Network loadedNet("big_network.bin", netMap);

    double* output2 = loadedNet.output(input.data());

    cout << "Loaded Network Output: ";
    for (int i = 0; i < loadedNet.getOutputSize(); ++i)
        cout << output2[i] << " ";
    cout << endl;

    // Compare
    if (compareOutputs(output1, output2, net.getOutputSize()))
        cout << "Outputs match for large network!" << endl;
    else
        cout << "Outputs do NOT match!" << endl;

    return 0;
}
