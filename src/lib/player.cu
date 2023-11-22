#include <iostream>
#include <string>

#include "player.h"

Player::Player(unsigned int num_empty_spaces) {
  switch (num_empty_spaces % 2) {
  case 0:
    player_type = Player::O;
    break;
  case 1:
    player_type = Player::X;
    break;
  }
}

Player::PlayerType Player::next_player_type() const {
  switch (player_type) {
  case Player::O:
    return Player::X;

  case Player::X:
    return Player::O;

  default:
    // std::cerr << "Player::Unknown(" + std::to_string(player_type) + ")";
    // ! Device code does not support exception handling.
    // throw std::invalid_argument("unknown player type");
    return Player::Error;
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
