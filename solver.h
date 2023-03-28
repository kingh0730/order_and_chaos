#pragma once

#include "primitive_value.h"
#include "position4x4.h"

class Solver
{
public:
    PrimitiveValue solve(const Position4x4 &position);
};
