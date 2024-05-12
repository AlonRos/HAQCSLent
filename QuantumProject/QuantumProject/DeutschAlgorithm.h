#pragma once
#include "Quregister.h"
#include "Utils.h"

Matrix2& U(int* f, int size);

int* generateBalancedFunction(int size);

int* generateConstantFunction(int size, int value);

bool isBalanced(int* f, int n);
