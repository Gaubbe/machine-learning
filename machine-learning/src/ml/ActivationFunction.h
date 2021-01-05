#pragma once
#include <math.h>


struct ActivationFunction {
	virtual double Function(double input) = 0;
	virtual double DerivedFunction(double input) = 0;
};

struct SigmoidActivationFunction : public ActivationFunction {
	double Function(double input) override { return 1 / (1 + exp(-input)); }
	double DerivedFunction(double input) override { return this->Function(input) * (1 - this->Function(input)); }
};

struct TanhActivationFunction : public ActivationFunction {
	double Function(double input) override { return tanh(input); }
	double DerivedFunction(double input) override { return 1 - (tanh(input) * tanh(input)); }
};

struct ReLUActivationFunction : public ActivationFunction {
	double Function(double input) override { return input < 0 ? 0 : input; }
	double DerivedFunction(double input) override { return input <= 0 ? 0 : 1; }
};