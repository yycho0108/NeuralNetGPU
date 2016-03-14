/*
 * Net.h
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */

#ifndef NET_H_
#define NET_H_

#include <cuda.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include "Layer.h"
#include "Utils.h"

class Net {
private:
	std::vector<int> topology;
	std::vector<Layer> layers;
	std::vector<std::vector<double>> weights;
	std::vector<std::vector<double>> wT;
public:
	Net(std::vector<int>);
	~Net();
	std::vector<double>& FF(std::vector<double> X);
	void BP(std::vector<double> X, std::vector<double> Y);
	void updatewT();
};

#endif /* NET_H_ */
