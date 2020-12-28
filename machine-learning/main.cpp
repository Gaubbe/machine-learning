#include <iostream>
#include <Eigen/Dense>

int main() 
{
	Eigen::MatrixXd m;
	m = Eigen::MatrixXd::Random(5, 5);
	std::cout << m << std::endl;
}