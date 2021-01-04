#pragma once
#include <vector>
#include <fstream>
#include <cstdint>

static std::vector<uint8_t> ReadFileIntoVector(const char* filePath, bool binary)
{
	std::ifstream file;
	if (binary) {
		file.open(filePath, std::ios::binary | std::ios::ate);
	} else {
		file.open(filePath, std::ios::ate);
	}
	auto size = file.tellg();
	file.seekg(0, std::ios::beg);
	char* array = new char[size];
	file.read(array, size);
	std::vector<uint8_t> result(array, array+size);
	return result;
}