#pragma once

#include <string>

enum Player
{
    Order = 0,
    Chaos = 1,
};

std::string format_player(const Player &player);