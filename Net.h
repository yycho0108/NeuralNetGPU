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
	std::vector<Layer*> layers;
	std::vector<double*> W;
	std::vector<double*> WT;
	std::vector<double*> dW;
public:
	Net(std::vector<int>);
	~Net();
	std::vector<double> FF(std::vector<double> X);
	void BP(std::vector<double> Y);
	void updateW();
};

#endif /* NET_H_ */
