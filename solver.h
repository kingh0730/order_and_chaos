#pragma once

#include "primitive_value.h"
#include "position4x4.h"
#include "recursive_value.h"

class Solver
{
private:
    GameResult solve_after_move(const Position4x4 &position, const Move &move);

public:
    GameResult solve(const Position4x4 &position);
};
