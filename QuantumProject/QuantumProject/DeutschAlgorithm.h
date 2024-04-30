#pragma once
#include "Quregister.h"
#include "Utils.h"

int* generateBalancedFunction(int size);

Matrix2& createMatrixFromFunction(int* f, int size);

bool isBalanced(int n, Matrix2& Uf);
