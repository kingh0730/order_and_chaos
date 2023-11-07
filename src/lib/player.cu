#include <iostream>
#include <string>

#include "player.h"

Player::PlayerType Player::next_player_type() const {
  switch (player_type) {
  case Player::O:
    return Player::X;

  case Player::X:
    return Player::O;

  default:
    std::cerr << "Player::Unknown(" + std::to_string(player_type) + ")";
    throw std::invalid_argument("unknown player type");
  }
}

std::string Player::format() const {
  switch (player_type) {
  case Player::O:
    return "Player::O";

  case Player::X:
    return "Player::X";

  default:
    std::cerr << "Player::Unknown(" + std::to_string(player_type) + ")";
    throw std::invalid_argument("unknown player type");
  }
}
