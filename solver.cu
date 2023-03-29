#include <algorithm>

#include "solver.h"

GameResult
Solver::solve_after_move(const Position4x4 &position, const Move &move)
{
    // naive approach
    const Position4x4 after_move = position.do_move(move);

    return this->solve(after_move);
}

GameResult
Solver::solve(const Position4x4 &position)
{
    auto pv = position.primitive_value();

    if (pv != PrimitiveValue::NotPrimitive)
    {
        return to_game_result(pv);
    }

    // children
    std::vector<Move> moves = position.generate_moves();
    std::vector<GameResult> grs = std::vector<GameResult>();
    grs.resize(moves.size());

    std::transform(
        moves.begin(), moves.end(), grs.begin(),
        [this, position](const Move &move)
        { return this->solve_after_move(position, move); });

    // recur step
    return game_result_recur_step(grs);
}
