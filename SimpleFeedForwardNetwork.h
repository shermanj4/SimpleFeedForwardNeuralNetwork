#pragma once
#include <math.h>
#include <vector>

using namespace std;

class SimpleFeedForwardNetwork
{
public:
	void initialize(int seed);
	
	int getEpoch();
	
	vector<double> getTrainingLosses();

	void train(const vector< vector< double > >& imagesTrainingSet, const vector< vector< double > >& imagesValidationSet,
		const vector<int>& labelsTrainingSet, vector<int>& labelsValidationSet ,size_t numEpochs);
		
	void validate(const vector< vector<double>>& imagesValidationSet, vector<int>& labelsValidationSet, vector<double>& allLosses);
	
	void test(const vector<vector<double>>& testing_images, const vector<int>& testing_labels);
	
	SimpleFeedForwardNetwork(double alpha, size_t hiddenLayerSize, size_t inputLayerSize, size_t numHiddenLayers, size_t outputLayerSize) :
		alpha(alpha), hiddenLayerSize(hiddenLayerSize), inputLayerSize(inputLayerSize), numHiddenLayers(numHiddenLayers), outputLayerSize(outputLayerSize) {}

private:
	vector<vector< vector< double > > > hiddenLayerWeights; // [layer][from][to]
	vector<vector< double > > outputLayerWeights;

	double alpha;
	int minEpoch;
	size_t hiddenLayerSize;
	size_t inputLayerSize;
	size_t numHiddenLayers;
	size_t outputLayerSize;

	inline double g(double x) {return 1.0 / (1.0 + exp(-x)); }
	inline double gprime(double y) {return y * (1 - y); }
	//uncomment for tanh
	//inline double g(double x) {return (exp(x) - exp(-x)) / (exp(x) + exp(-x)); }
	//inline double gprime(double y) {return 1 - (tanh(y)*tanh(y)); }
};
