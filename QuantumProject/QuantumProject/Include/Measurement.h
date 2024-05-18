#pragma once

#ifndef Measurement_H
#define Measurement_H

#include "../Include/Quregister.h"
#include <vector>

using namespace std;

// the basis is orthonormal
int measure(Quregister& reg, vector<Quregister> basis);

int measureComputational(Quregister& reg);

int measureComputational(Quregister& reg, int i, int j);

int measureInSubSpaces(Quregister& reg, vector<vector<Quregister>> bases);

#endif