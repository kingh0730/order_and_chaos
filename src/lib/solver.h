#pragma once

#include <map>

#include "primitive_value.h"
#include "position4x4.h"
#include "recursive_value.h"

class Solver
{
private:
    std::map<Position4x4, GameResult> memoized;

private:
    std::vector<GameResult> solve_children(
        const Position4x4 &position);

    GameResult solve_one_child(
        const Position4x4 &position, const Move &move);

public:
    Solver() : memoized(std::map<Position4x4, GameResult>()) {}

    GameResult solve_not_memoized(const Position4x4 &position);
    GameResult solve(const Position4x4 &position);
};
