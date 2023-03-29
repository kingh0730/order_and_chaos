#include <iostream>
#include <algorithm>

#include "solver.h"

std::vector<GameResult>
Solver::solve_children(const Position4x4 &position)
{
    std::vector<Move> moves = position.generate_moves();

    std::vector<GameResult> grs = std::vector<GameResult>();
    grs.resize(moves.size());

    std::transform(
        moves.begin(), moves.end(), grs.begin(),
        [this, position](const Move &move)
        { return this->solve_one_child(position, move); });

    return grs;
}

GameResult
Solver::solve_one_child(const Position4x4 &position, const Move &move)
{
    // TODO There could clever ways

    // naive approach
    const Position4x4 child = position.do_move(move);

    return this->solve(child);
}

GameResult
Solver::solve(const Position4x4 &position)
{
    // FIXME delete shortcut and print
    // Order has to win in 6 steps or less
    if (position.get_num_spaces_remain() <= 10)
    {
        return GameResult::GameLose;
    }
#ifdef MAIN_TEST_PRINT
    if (position.primitive_value() != PrimitiveValue::NotPrimitive)
    {
        std::cout << position.format_pretty() << std::endl;
    }
#endif

    // Primitive value
    auto pv = position.primitive_value();

    // If not primitive
    if (pv != PrimitiveValue::NotPrimitive)
    {
        return to_game_result(pv);
    }

    // children
    std::vector<GameResult> grs = solve_children(position);

    // recursive step
    return game_result_recur_step(grs);
}
