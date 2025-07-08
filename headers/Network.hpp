#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include <string>
#include "../headers/Layer.hpp"

using namespace std;

class Network
{
private:
    int* netMap;
    bool made_here;
    int numOfLayers;
    Layer* Layers;

    void outForward(Layer& curr, Layer& prev);

public:
    Network(int numOfLayers,int* netMap);
    Network(string FilePath);

    bool store(string FilePath);

    double* output(double* input);
    int getOutputSize();
    int getNumLayers();
    Layer* getLayers();

    double sigmoid(double Z);
    double sigmoidDrev(double Z);

    ~Network();
};


#endif // NETWORK_HPP_INCLUDED
