#pragma once

#include <iostream>

enum Player
{
    Order,
    Chaos,
};

void print_player(Player player)
{
    switch (player)
    {
    case Player::Order:
        std::cout << "Player::Order";
    case Player::Chaos:
        std::cout << "Player::Chaos";
    default:
        std::cout << "Player::Unknown";
    }
}
