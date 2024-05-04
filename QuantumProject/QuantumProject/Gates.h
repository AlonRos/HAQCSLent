#ifndef Gates_H
#define Gates_h

#include "Matrix2.h"

extern Matrix2& bitFlip;
extern Matrix2& hadamard;
extern Matrix2& CNOT;

Matrix2& phaseShift(double phi);

#endif