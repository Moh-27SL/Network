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

int main() {
    
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

    return 0;
}
