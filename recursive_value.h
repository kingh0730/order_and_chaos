#pragma once

#include <vector>

#include "primitive_value.h"

enum GameResult
{
    Undecided,
    GameWin,
    GameLose,
    GameTie,
    GameDraw,
};

GameResult to_game_result(const PrimitiveValue &pv);
GameResult game_result_recur_step(const std::vector<GameResult> &children);

std::string format_game_result(const GameResult &gr);

// TODO
class GameResultWithRmt
{
private:
    GameResult result;
    int remoteness;
};
