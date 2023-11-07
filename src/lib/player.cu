#include <iostream>
#include <string>

#include "player.h"

Player Player::next_player() const {
  switch (player) {
  case Player::O:
    return Player::X;

  case Player::X:
    return Player::O;

  default:
    std::cerr << "Player::Unknown(" + std::to_string(player) + ")";
    throw std::invalid_argument("unknown player type");
  }
}

std::string Player::format() const {
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
