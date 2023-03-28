#pragma once

enum PrimitiveValue
{
    NotPrimitive = 0,
    Win = 1,
    Lose = 2,
    Tie = 3,
};

std::string format_primitive_value(PrimitiveValue &pv);
