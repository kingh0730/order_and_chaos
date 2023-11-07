#pragma once

#include <vector>

#include "primitive_value.h"

// Keep small as map value
enum GameResult : char
{
    Undecided,
    GameWin,
    GameLose,
    GameTie,
    GameDraw,
};

GameResult to_game_result(const PrimitiveValue &pv);

// Recursive step
GameResult game_result_recur_step(const std::vector<GameResult> &children);

// Utils
GameResult win_lose_switch(const GameResult &gr);

// Formatting
std::string format_game_result(const GameResult &gr);

// TODO Remoteness
class GameResultWithRmt
{
private:
    GameResult result;
    int remoteness;
};
