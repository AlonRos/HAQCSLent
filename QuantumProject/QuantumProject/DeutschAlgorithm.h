#pragma once
#include "Quregister.h"
#include "Utils.h"

int* generateBalancedFunction(int size);

int* generateConstantFunction(int size, int value);

bool isBalanced(int n, Matrix2& Uf);
