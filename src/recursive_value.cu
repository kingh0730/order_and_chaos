#include <algorithm>
#include <iostream>

#include "recursive_value.h"

GameResult to_game_result(const PrimitiveValue &pv)
{
    switch (pv)
    {
    case PrimitiveValue::NotPrimitive:
        return GameResult::Undecided;

    case PrimitiveValue::Win:
        return GameResult::GameWin;

    case PrimitiveValue::Lose:
        return GameResult::GameLose;

    case PrimitiveValue::Tie:
        return GameResult::GameTie;

    default:
        std::cerr << "Unknown primitive value: " << pv << std::endl;
        throw std::invalid_argument("Unknown primitive value");
    }
}

GameResult game_result_recur_step(const std::vector<GameResult> &children)
{
    bool any_child_lose = std::any_of(
        children.begin(), children.end(),
        [](const GameResult &gv)
        { return gv == GameResult::GameLose; });

    if (any_child_lose)
    {
        return GameResult::GameWin;
    }

    // FIXME Tie does not appear in order_and_chaos

    return GameResult::GameLose;
}

std::string format_game_result(const GameResult &gr)
{
    switch (gr)
    {
    case GameResult::Undecided:
        return "GameResult::Undecided";

    case GameResult::GameWin:
        return "GameResult::GameWin";

    case GameResult::GameLose:
        return "GameResult::GameLose";

    case GameResult::GameTie:
        return "GameResult::GameTie";

    case GameResult::GameDraw:
        return "GameResult::GameDraw";

    default:
        std::cerr << "GameResult::Unknown(" + std::to_string(gr) + ")";
        throw std::invalid_argument("unknown GameResult type");
    }
}
