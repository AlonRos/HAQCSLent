#pragma once

#ifndef Measurement_H
#define Measurement_H

#include "Quregister.h"
#include <vector>

using namespace std;

// the basis is orthonormal
int measure(Quregister& reg, vector<Quregister> basis);

int measureComputational(Quregister& reg);


#endif