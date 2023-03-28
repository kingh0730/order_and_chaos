#include "player.h"

void print_player(Player &player)
{
    switch (player)
    {
    case Player::Order:
        std::cout << "Player::Order";
        break;

    case Player::Chaos:
        std::cout << "Player::Chaos";
        break;

    default:
        std::cout << "Player::Unknown\n";
        std::cout << "value: " << player << std::endl;
        throw std::invalid_argument("unknown player type");
    }
}
