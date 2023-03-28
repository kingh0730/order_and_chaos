#include "solver.h"

GameResult Solver::solve(const Position4x4 &position)
{
    auto pv = position.primitive_value();

    if (pv != PrimitiveValue::NotPrimitive)
    {
        return to_game_result(pv);
    }

    // TODO

    return GameResult::Undecided;
}
