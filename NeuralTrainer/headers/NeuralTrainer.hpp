#ifndef NEURALTRAINER_HPP
#define NEURALTRAINER_HPP
#include "../../headers/Network.hpp"
#define DECAY_RATE 0.5
#define DECAY_STEP 100
#include <math.h>

class NeuralTrainer
{
    public:
        NeuralTrainer(Network* network);
        void setInputs(double** inputs, unsigned int numOfInputs);
        virtual ~NeuralTrainer();
        void updateBiases();
        void updateWeights();
        void train(unsigned int epochs, double eta, double lambda, unsigned int miniBatchSize);
        bool storeLoss(string FilePath);

        double** GetoutErr() { return outErr; }
        double* GetAcc() {return accuracy; }
        double* GetLoss() {return loss; }
        void SetoutErr(double** val) { outErr = val; }
        double** GetreqOut() { return reqOut; }
        void SetreqOut(double** val) { reqOut = val; }
        double Geteta() { return eta; }
        void Seteta(double val) { eta = val; }
        double Getlambda() { return lambda; }
        void Setlambda(double val) { lambda = val; }
        unsigned int GetminiBatSize() { return miniBatSize; }
        void SetminiBatSize(unsigned int val) { miniBatSize = val; }
        double getDecayedEta(double baseEta, int epoch)
        {
            double decayRate = DECAY_RATE;
            int decayStep = DECAY_STEP;
            return baseEta * pow(decayRate, epoch / decayStep);
        }

    protected:

    private:

        Network* network;
        double** outErr;
        double** reqOut;
        double** inputs;
        double* loss;
        double* accuracy;
        double eta;
        double lambda;
        unsigned int miniBatSize;
        unsigned int numOfInputs;
        unsigned int numOfLayers;
        unsigned int epochs;

        void outErrLastLayer(int reqOutIndex);
        void outErrLayer(unsigned int LayerNum, int reqOutIndex);
};

#endif // NEURALTRAINER_HPP
