#pragma once

#include <string>

enum Player {
  O = 0,
  X = 1,
};

std::string format_player(const Player &player);
