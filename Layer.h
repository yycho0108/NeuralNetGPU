/*
 * Layer.h
 *
 *  Created on: Mar 13, 2016
 *      Author: jamiecho
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <vector>

class Layer {
private:
	std::vector<double> I; //input
	std::vector<double> O; //output
	std::vector<double> G; //gradient

public:
	Layer(int n);
	~Layer();
	std::vector<double>& transfer(std::vector<double>);

	void setI(std::vector<double>);
	void setO(std::vector<double>);
	void setG(std::vector<double>);

	std::vector<double>& getI();
	std::vector<double>& getO();
	std::vector<double>& getG();
};

#endif /* LAYER_H_ */
