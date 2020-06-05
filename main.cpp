#include <vector>
#include <iostream>
#include <iomanip> 

#include "MNIST_reader.h"
#include "SimpleFeedForwardNetwork.h"
using namespace std;
int main()
{
	double alpha = 0.8;   // learning rate
	size_t inputLayerSize = 784;
	size_t hiddenLayerSize = 42;
	size_t numEpochs = 3;
	size_t numHiddenLayers = 2;
	size_t outputLayerSize = 10;
	
	int seed = 0; // random seed for the network initialization
	
	string filename = "/Users/joshuasherman/Documents/CS360/code/task3/train-images-idx3-ubyte";
	//load MNIST images
	vector <vector< int> > training_images;
	loadMnistImages(filename, training_images);
	cout << "Number of images: " << training_images.size() << endl;
	cout << "Image size: " << training_images[0].size() << endl;
	vector< vector<double> > trainingSet(5000);
	vector< vector<double> > validationSet(2000);
	for(int i = 0; i < trainingSet.size()+validationSet.size(); i++){
		for(int j = 0; j < training_images[0].size(); j++){
			if(i < 5000){
				trainingSet[i].push_back(double(training_images[i][j]) / double(255));
			}else{
				validationSet[i-5000].push_back(double(training_images[i][j]) / double(255));\
			}
		}
	}
	filename = "/Users/joshuasherman/Documents/CS360/code/task3/train-labels-idx1-ubyte";
	//load MNIST labels
	vector<int> training_labels;
	vector<int> trainingLabelsSet(5000);
	vector<int> validationLabelsSet(2000);
	loadMnistLabels(filename, training_labels);
	cout << "Number of labels: " << training_labels.size() << endl;
	for(int j = 0; j < 7000; j++){
		if(j < 5000){
			trainingLabelsSet[j] = (training_labels[j]);
		}else{
			validationLabelsSet[j] = (training_labels[j]);
		}
	}
	
	SimpleFeedForwardNetwork nn(alpha, hiddenLayerSize, inputLayerSize, numHiddenLayers, outputLayerSize);
	nn.initialize(seed);
	nn.train(trainingSet, validationSet, trainingLabelsSet, validationLabelsSet, numEpochs);
	
	filename = "/Users/joshuasherman/Documents/CS360/code/task3/t10k-images-idx3-ubyte";
	vector<vector<int>> testing_imageInt;
	loadMnistImages(filename, testing_imageInt);
	vector<vector<double>> testing_images(testing_imageInt.size());
	for(int i = 0; i < testing_images.size(); i++){
		for(int j = 0; j < testing_imageInt[0].size(); j++){
			testing_images[i].push_back(double(testing_imageInt[i][j]) / double(255));
		}
	}
	filename = "/Users/joshuasherman/Documents/CS360/code/task3/t10k-labels-idx1-ubyte";
	vector<int> testing_labels;
	loadMnistLabels(filename, testing_labels);
	nn.initialize(seed);
	nn.train(trainingSet, validationSet, trainingLabelsSet, validationLabelsSet, nn.getEpoch()+1);
	nn.test(testing_images, testing_labels);
	
	
	return 0;
}