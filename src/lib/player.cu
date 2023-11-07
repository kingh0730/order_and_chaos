#include <iostream>
#include <string>

#include "_player.h"

std::string format_player(const Player &player) {
  switch (player) {
  case Player::O:
    return "Player::O";

  case Player::X:
    return "Player::X";

  default:
    std::cerr << "Player::Unknown(" + std::to_string(player) + ")";
    throw std::invalid_argument("unknown player type");
  }
}
