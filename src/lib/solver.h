#pragma once

#include <map>

#include "primitive_value.h"
#include "position4x4.h"
#include "recursive_value.h"

class Solver
{
    // TODO memoized states
    // Keep small as map value
    enum MemoizedState1 : char
    {
        MemoizedState1_1,
        MemoizedState1_2,
    };

    // TODO memoized states
    // Keep small as map value
    enum MemoizedState2 : char
    {
        MemoizedState2_1,
        MemoizedState2_2,
    };

private:
    std::map<Position4x4, GameResult> memoized_game_results;

    GameResult memoized_to_game_result(
        const Position4x4 &cur_position, const MemoizedGameResult &mgr);

    MemoizedGameResult game_result_to_memoized(
        const Position4x4 &cur_position, const GameResult &gr);

private:
    std::vector<GameResult> solve_children(
        const Position4x4 &position);

    GameResult solve_one_child(
        const Position4x4 &position, const Move &move);

public:
    Solver() : memoized(std::map<Position4x4, MemoizedGameResult>()) {}

    GameResult solve_not_memoized(const Position4x4 &position);
    GameResult solve(const Position4x4 &position);
};
