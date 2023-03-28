#include <iostream>
#include <string>

#include "player.h"

std::string format_player(Player &player)
{
    switch (player)
    {
    case Player::Order:
        return "Player::Order";

    case Player::Chaos:
        return "Player::Chaos";

    default:
        std::cerr << "Player::Unknown(" + std::to_string(player) + ")";
        throw std::invalid_argument("unknown player type");
    }
}
