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
        MemoizedState1_0,
        MemoizedState1_1,
    };

    // TODO memoized states
    // Keep small as map value
    enum MemoizedState2 : char
    {
        MemoizedState2_0,
        MemoizedState2_1,
    };

private:
    // TODO This could be a pointer to avoid copying
    std::map<Position4x4, GameResult> memoized_game_results;

    std::map<Position4x4, MemoizedState1> memoized_states1;
    std::map<Position4x4, MemoizedState2> memoized_states2;

    // TODO memoized states
    GameResult memoized_states_to_game_result(
        const Position4x4 &cur_position,
        const MemoizedState1 &ms1,
        const MemoizedState2 &ms2);

    MemoizedState1 game_result_to_memoized_state1(
        const Position4x4 &cur_position, const GameResult &gr);

    MemoizedState2 game_result_to_memoized_state2(
        const Position4x4 &cur_position, const GameResult &gr);

private:
    std::vector<GameResult> solve_children(
        const Position4x4 &position);

    GameResult solve_one_child(
        const Position4x4 &position, const Move &move);

public:
    Solver() : memoized_game_results(std::map<Position4x4, GameResult>()) {}
    Solver(std::map<Position4x4, GameResult> &load)
        : memoized_game_results(load) {}

    GameResult solve_not_memoized(const Position4x4 &position);
    GameResult solve(const Position4x4 &position);

    std::string format() const;
};
