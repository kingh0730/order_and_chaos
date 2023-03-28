#pragma once

enum GameResult
{
    Win,
    Lose,
    Tie,
    Draw,
};

// TODO
class GameResultWithRmt
{
private:
    GameResult result;
    int remoteness;
};
