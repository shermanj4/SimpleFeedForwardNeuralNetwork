#include "SimpleFeedForwardNetwork.h"
#include <iostream>
#include <random>
#include <iomanip>      // std::setprecision

int SimpleFeedForwardNetwork::getEpoch(){
	return minEpoch;
}

void SimpleFeedForwardNetwork::initialize(int seed)
{
	srand(seed);
	hiddenLayerWeights.resize(numHiddenLayers);
	hiddenLayerWeights[0].resize(inputLayerSize);
	for (size_t i = 0; i < inputLayerSize; i++)
	{
		hiddenLayerWeights[0][i].resize(hiddenLayerSize);
		for (size_t j = 0; j < hiddenLayerSize; j++)
		{
			double f = (double)rand() / RAND_MAX;
			hiddenLayerWeights[0][i][j] = (0 + f * (1 - 0))-0.5;	// This network cannot learn if the initial weights are set to zero.
		}
	}
	//loop through and assign weights between hidden layers
	for(size_t k = 1; k < numHiddenLayers; k++){
		hiddenLayerWeights[k].resize(hiddenLayerSize);
		for (size_t i = 0; i < hiddenLayerSize; i++)
		{
			hiddenLayerWeights[k][i].resize(hiddenLayerSize);
			for (size_t j = 0; j < hiddenLayerSize; j++)
			{
				double f = (double)rand() / RAND_MAX;
				hiddenLayerWeights[k][i][j] = (0 + f * (1 - 0))-0.5; 	// This network cannot learn if the initial weights are set to zero.
			}
		}
	}
	outputLayerWeights.resize(hiddenLayerSize);
	for(size_t i = 0; i < hiddenLayerSize; i++){
		outputLayerWeights[i].resize(outputLayerSize);
		for (size_t j = 0; j < outputLayerSize; j++)
		{
			double f = (double)rand() / RAND_MAX;
			outputLayerWeights[i][j] = (0 + f * (1 - 0))-0.5; 	// This network cannot learn if the initial weights are set to zero.
		}
	}
}

void SimpleFeedForwardNetwork::validate(const vector< vector<double>>& imagesValidationSet, vector<int>& labelsValidationSet, vector<double>& allLosses){
	double sumOfCorrect = 0;
	double total = 0;
	double predictionAccuracy = 0;
	double testLoss = 0;
	for (size_t example = 0; example < imagesValidationSet.size(); example++){
			// propagate the inputs forward to compute the outputs 
			vector< double > activationInput(inputLayerSize); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			vector<vector< double > > activationHidden(numHiddenLayers);
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
			{
				activationInput[inputNode] = imagesValidationSet[example][inputNode];
			}
			activationHidden[0].resize(hiddenLayerSize);
			// calculate activations of hidden layers (for now, just one hidden layer)
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				double inputToHidden = 0;
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
				{
					inputToHidden += hiddenLayerWeights[0][inputNode][hiddenNode] * activationInput[inputNode];
				}
				activationHidden[0][hiddenNode] = g(inputToHidden);
			}
			//calculate activations for the rest of the hidden layers
			for(size_t k = 1; k < numHiddenLayers; k++){
				activationHidden[k].resize(hiddenLayerSize);
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					double hiddenToHidden = 0;
					for (size_t inputHiddenNode = 0; inputHiddenNode < hiddenLayerSize; inputHiddenNode++)
					{
						hiddenToHidden += hiddenLayerWeights[k][inputHiddenNode][hiddenNode] * activationHidden[k-1][inputHiddenNode];
					}
					activationHidden[k][hiddenNode] = g(hiddenToHidden);
				}
			}
			//more than one output node.
			vector<double> activationOutput(outputLayerSize);
			for(size_t outputNode = 0; outputNode < outputLayerSize; outputNode++){
				double inputAtOutput = 0;
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					inputAtOutput += outputLayerWeights[hiddenNode][outputNode] * activationHidden[numHiddenLayers-1][hiddenNode];
					
				}
				activationOutput[outputNode] = g(inputAtOutput);
			}	
			for(int i = 0; i < outputLayerSize; i++){
				//cout << " " << std::setprecision(1) << activationOutput[i];
			}
			//calculating errors and loss
			vector<double> errorOfOutputNode(outputLayerSize);
			double encoding[10];
			for(int i = 0; i < 10; i++){
				if(i == labelsValidationSet[example]){
					encoding[i] = 1.;
					if(activationOutput[i] == *max_element(activationOutput.begin(), activationOutput.end())){
						sumOfCorrect++;
					}
				}else{
					encoding[i] = 0.;
				}
			}
			//uncomment for binary encoding
			// double encoding[4];
// 			if(labelsValidationSet[example] == 0){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 1;
// 				if(activationOutput[1] > activationOutput[0] && activationOutput[2] > activationOutput[0] && activationOutput[3] > activationOutput[0]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 1){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[0] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 2){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[1] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 3){
// 				encoding[0] = 1;
// 				encoding[1] = 1;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[0] > activationOutput[2] && activationOutput[0] > activationOutput[3] && activationOutput[1] > activationOutput[2] && activationOutput[1] > activationOutput[3]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 4){
// 				encoding[0] = 0;
// 				encoding[1] = 0;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[2] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 5){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[0] > activationOutput[1] && activationOutput[0] > activationOutput[3] && activationOutput[2] > activationOutput[1] && activationOutput[2] > activationOutput[3]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 6){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[1] > activationOutput[0] && activationOutput[1] > activationOutput[3] && activationOutput[2] > activationOutput[0] && activationOutput[2] > activationOutput[3]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 7){
// 				encoding[0] = 1;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[0] > activationOutput[3] && activationOutput[1] > activationOutput[3] && activationOutput[2] > activationOutput[3]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 8){
// 				encoding[0] = 0;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 1;
// 				if(activationOutput[3] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsValidationSet[example] == 9){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 1;
// 				if(activationOutput[0] > activationOutput[1] && activationOutput[0] > activationOutput[2] && activationOutput[3] > activationOutput[1] && activationOutput[3] > activationOutput[2]){
// 					sumOfCorrect++;
// 				}
// 			}
			total++;
			for(size_t i = 0; i < outputLayerSize; i++){
				testLoss += ((encoding[i] - activationOutput[i])*(encoding[i] - activationOutput[i]));
				//cout << " " << errorOfOutputNode[0];
			}
		}
		//cout << testLoss << endl;
		allLosses.push_back(testLoss);
		predictionAccuracy = sumOfCorrect/total;
		//cout << "Validation Accuracy: " << predictionAccuracy << " ";
}

void SimpleFeedForwardNetwork::train(const vector< vector< double > >& imagesTrainingSet, const vector< vector< double > >& imagesValidationSet,
		const vector<int>& labelsTrainingSet, vector<int>& labelsValidationSet ,size_t numEpochs)
{
	size_t trainingexamples = imagesTrainingSet.size();
	// train the network
	vector<double> allLosses;
	for (size_t epoch = 0; epoch < numEpochs; epoch++)
	{
		// print
		double sumOfCorrect = 0;
		double total = 0;
		double predictionAccuracy = 0;
		cout << "epoch = " << epoch << " ";
		validate(imagesValidationSet, labelsValidationSet, allLosses); 
		double testLoss = 0;
		for (size_t example = 0; example < trainingexamples; example++)
		{
			// propagate the inputs forward to compute the outputs
			vector< double > activationInput(inputLayerSize); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			vector<vector< double > > activationHidden(numHiddenLayers);
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
			{
				activationInput[inputNode] = imagesTrainingSet[example][inputNode];
			}
			activationHidden[0].resize(hiddenLayerSize);
			// calculate activations of hidden layers (for now, just one hidden layer)
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				double inputToHidden = 0;
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
				{
					inputToHidden += hiddenLayerWeights[0][inputNode][hiddenNode] * activationInput[inputNode];
				}
				activationHidden[0][hiddenNode] = g(inputToHidden);
			}
			//calculate activations for the rest of the hidden layers
			for(size_t k = 1; k < numHiddenLayers; k++){
				activationHidden[k].resize(hiddenLayerSize);
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					double hiddenToHidden = 0;
					for (size_t inputHiddenNode = 0; inputHiddenNode < hiddenLayerSize; inputHiddenNode++)
					{
						hiddenToHidden += hiddenLayerWeights[k][inputHiddenNode][hiddenNode] * activationHidden[k-1][inputHiddenNode];
					}
					activationHidden[k][hiddenNode] = g(hiddenToHidden);
				}
			}
			//more than one output node.
			vector<double> activationOutput(outputLayerSize);
			for(size_t outputNode = 0; outputNode < outputLayerSize; outputNode++){
				double inputAtOutput = 0;
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					inputAtOutput += outputLayerWeights[hiddenNode][outputNode] * activationHidden[numHiddenLayers-1][hiddenNode];
					
				}
				activationOutput[outputNode] = g(inputAtOutput);
			}	
			//cout << endl;
			//cout << labelsTrainingSet[example] << endl;
			for(int i = 0; i < outputLayerSize; i++){
				//cout << " " << std::setprecision(1) << activationOutput[i];
			}
			//cout << " ";
			//calculating errors and loss
			vector<double> errorOfOutputNode(outputLayerSize);
			double encoding[10];
			for(int i = 0; i < 10; i++){
				if(i == labelsTrainingSet[example]){
					encoding[i] = 1.;
					if(activationOutput[i] == *max_element(activationOutput.begin(), activationOutput.end())){
						sumOfCorrect++;
					}
				}else{
					encoding[i] = 0.;
				}
			}
			//uncomment for binary encoding
			// double encoding[4];
// 			if(labelsTrainingSet[example] == 0){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 1;
// 				if(activationOutput[1] > activationOutput[0] && activationOutput[2] > activationOutput[0] && activationOutput[3] > activationOutput[0]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 1){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[0] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 2){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[1] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 3){
// 				encoding[0] = 1;
// 				encoding[1] = 1;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if((activationOutput[0] > activationOutput[2] && activationOutput[0] > activationOutput[3]) && (activationOutput[1] > activationOutput[2] && activationOutput[1] > activationOutput[3])){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 4){
// 				encoding[0] = 0;
// 				encoding[1] = 0;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[2] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 5){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if((activationOutput[0] > activationOutput[1] && activationOutput[0] > activationOutput[3]) && (activationOutput[2] > activationOutput[1] && activationOutput[2] > activationOutput[3])){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 6){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if((activationOutput[1] > activationOutput[0] && activationOutput[1] > activationOutput[3]) && (activationOutput[2] > activationOutput[0] && activationOutput[2] > activationOutput[3])){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 7){
// 				encoding[0] = 1;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[0] > activationOutput[3] && activationOutput[1] > activationOutput[3] && activationOutput[2] > activationOutput[3]){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 8){
// 				encoding[0] = 0;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 1;
// 				if(activationOutput[3] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(labelsTrainingSet[example] == 9){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 1;
// 				if((activationOutput[0] > activationOutput[1] && activationOutput[0] > activationOutput[2]) && (activationOutput[3] > activationOutput[1] && activationOutput[3] > activationOutput[2])){
// 					sumOfCorrect++;
// 				}
// 			}
			total++;
			
			
			for(size_t i = 0; i < outputLayerSize; i++){
				errorOfOutputNode[i] = gprime(activationOutput[i]) * (encoding[i] - activationOutput[i]);
				testLoss += (encoding[i] - activationOutput[i])*(encoding[i] - activationOutput[i]);
				//cout << " " << encoding[i];
			}
			//cout << testLoss;
			// Calculating error of hidden layer. Special calculation since we only have one output node; i.e. no summation over next layer nodes
			// Also adjusting weights of output layer
			vector<vector< double > > errorOfHiddenNode(numHiddenLayers);
			for(int hiddenLayer = numHiddenLayers - 1; hiddenLayer >= 0; hiddenLayer--){
				errorOfHiddenNode[hiddenLayer].resize(hiddenLayerSize);
				if(hiddenLayer == numHiddenLayers - 1){
					for(size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++){
						for(size_t outputNode = 0; outputNode < outputLayerSize; outputNode++){
							errorOfHiddenNode[hiddenLayer][hiddenNode] += outputLayerWeights[hiddenNode][outputNode] * errorOfOutputNode[outputNode];
						}
						errorOfHiddenNode[hiddenLayer][hiddenNode] *= gprime(activationHidden[numHiddenLayers-1][hiddenNode]);
					}
				}else{		
					for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
					{
						for(size_t node = 0; node < hiddenLayerSize; node++){
							errorOfHiddenNode[hiddenLayer][hiddenNode] += hiddenLayerWeights[hiddenLayer][hiddenNode][node] * errorOfHiddenNode[hiddenLayer+1][node];
						}
						errorOfHiddenNode[hiddenLayer][hiddenNode] *= gprime(activationHidden[hiddenLayer][hiddenNode]);	
					}
				}
			}
			//adjusting weights
			//adjusting weights at output layer
			for(size_t outputNode = 0; outputNode < outputLayerSize; outputNode++){
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					outputLayerWeights[hiddenNode][outputNode] += alpha * activationHidden[numHiddenLayers-1][hiddenNode] * errorOfOutputNode[outputNode];
				}
			}
			// Adjusting weights at 1st hidden layer.
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
				{
					hiddenLayerWeights[0][inputNode][hiddenNode] += alpha * activationInput[inputNode] * errorOfHiddenNode[0][hiddenNode];
				}
			}
			
			for(size_t k = 1; k < numHiddenLayers; k++){
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					for (size_t inputHiddenNode = 0; inputHiddenNode < hiddenLayerSize; inputHiddenNode++)
					{
						hiddenLayerWeights[k][inputHiddenNode][hiddenNode] += alpha * activationHidden[k-1][inputHiddenNode] * errorOfHiddenNode[k][hiddenNode];
					}
					
				}
			}
			
			
		//cout << " " << labelsTrainingSet[example];	
		}
		predictionAccuracy = sumOfCorrect/total;
		cout << "Training Accuracy: " << predictionAccuracy << " ";
		cout << endl;
	}
	double minValidationLoss = allLosses[0];
	int index = 0;
	for(int i = 0; i < allLosses.size(); i++){
		if(allLosses[i] < minValidationLoss){
			minValidationLoss = allLosses[i];
			index = i;
		}
	}
	this->minEpoch = index;
	cout << "Epoch " << this->minEpoch << " has the lowest validation loss: " << minValidationLoss << endl;
	return;
}

void SimpleFeedForwardNetwork::test(const vector<vector<double>>& testing_images, const vector<int>& testing_labels){
	double sumOfCorrect = 0;
	double total = 0;
	double predictionAccuracy = 0;
	for (size_t example = 0; example < testing_images.size(); example++){
			// propagate the inputs forward to compute the outputs 
			vector< double > activationInput(inputLayerSize); // We store the activation of each node (over all input and hidden layers) as we need that data during back propagation.			
			vector<vector< double > > activationHidden(numHiddenLayers);
			for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++) // initialize input layer with training data
			{
				activationInput[inputNode] = testing_images[example][inputNode];
			}
			activationHidden[0].resize(hiddenLayerSize);
			// calculate activations of hidden layers (for now, just one hidden layer)
			for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
			{
				double inputToHidden = 0;
				for (size_t inputNode = 0; inputNode < inputLayerSize; inputNode++)
				{
					inputToHidden += hiddenLayerWeights[0][inputNode][hiddenNode] * activationInput[inputNode];
				}
				activationHidden[0][hiddenNode] = g(inputToHidden);
			}
			//calculate activations for the rest of the hidden layers
			for(size_t k = 1; k < numHiddenLayers; k++){
				activationHidden[k].resize(hiddenLayerSize);
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					double hiddenToHidden = 0;
					for (size_t inputHiddenNode = 0; inputHiddenNode < hiddenLayerSize; inputHiddenNode++)
					{
						hiddenToHidden += hiddenLayerWeights[k][inputHiddenNode][hiddenNode] * activationHidden[k-1][inputHiddenNode];
					}
					activationHidden[k][hiddenNode] = g(hiddenToHidden);
				}
			}
			//more than one output node.
			vector<double> activationOutput(outputLayerSize);
			for(size_t outputNode = 0; outputNode < outputLayerSize; outputNode++){
				double inputAtOutput = 0;
				for (size_t hiddenNode = 0; hiddenNode < hiddenLayerSize; hiddenNode++)
				{
					inputAtOutput += outputLayerWeights[hiddenNode][outputNode] * activationHidden[numHiddenLayers-1][hiddenNode];
					
				}
				activationOutput[outputNode] = g(inputAtOutput);
			}	
			for(int i = 0; i < outputLayerSize; i++){
				//cout << " " << std::setprecision(1) << activationOutput[i];
			}
			//calculating errors and loss
			vector<double> errorOfOutputNode(outputLayerSize);
			double encoding[10];
			for(int i = 0; i < 10; i++){
				if(i == testing_labels[example]){
					encoding[i] = 1.;
					if(activationOutput[i] == *max_element(activationOutput.begin(), activationOutput.end())){
						sumOfCorrect++;
					}
				}else{
					encoding[i] = 0.;
				}
			}
			//uncomment for binary encoding
			// double encoding[4];
// 			if(testing_labels[example] == 0){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 1;
// 				if(activationOutput[1] > activationOutput[0] && activationOutput[2] > activationOutput[0] && activationOutput[3] > activationOutput[0]){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 1){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[0] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 2){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if(activationOutput[1] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 3){
// 				encoding[0] = 1;
// 				encoding[1] = 1;
// 				encoding[2] = 0;
// 				encoding[3] = 0;
// 				if((activationOutput[0] > activationOutput[2] && activationOutput[0] > activationOutput[3]) && (activationOutput[1] > activationOutput[2] && activationOutput[1] > activationOutput[3])){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 4){
// 				encoding[0] = 0;
// 				encoding[1] = 0;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[2] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 5){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if((activationOutput[0] > activationOutput[1] && activationOutput[0] > activationOutput[3]) && (activationOutput[2] > activationOutput[1] && activationOutput[2] > activationOutput[3])){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 6){
// 				encoding[0] = 0;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if((activationOutput[1] > activationOutput[0] && activationOutput[1] > activationOutput[3]) && (activationOutput[2] > activationOutput[0] && activationOutput[2] > activationOutput[3])){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 7){
// 				encoding[0] = 1;
// 				encoding[1] = 1;
// 				encoding[2] = 1;
// 				encoding[3] = 0;
// 				if(activationOutput[0] > activationOutput[3] && activationOutput[1] > activationOutput[3] && activationOutput[2] > activationOutput[3]){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 8){
// 				encoding[0] = 0;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 1;
// 				if(activationOutput[3] == *max_element(activationOutput.begin(), activationOutput.end())){
// 					sumOfCorrect++;
// 				}
// 			}else if(testing_labels[example] == 9){
// 				encoding[0] = 1;
// 				encoding[1] = 0;
// 				encoding[2] = 0;
// 				encoding[3] = 1;
// 				if((activationOutput[0] > activationOutput[1] && activationOutput[0] > activationOutput[2]) && (activationOutput[3] > activationOutput[1] && activationOutput[3] > activationOutput[2])){
// 					sumOfCorrect++;
// 				}
// 			}
			total++;
		}
		predictionAccuracy = sumOfCorrect/total;
		cout << "Testing Accuracy: " << predictionAccuracy << " ";
}
