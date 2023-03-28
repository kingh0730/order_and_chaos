#pragma once

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

// TODO
class GameResultWithRmt
{
private:
    GameResult result;
    int remoteness;
};
