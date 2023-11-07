#pragma once

#include "board.h"
#include "player.h"
#include <string>

class Position {
private:
  Player player;
  Board board;

public:
  Position(Player player, Board board) : player(player), board(board) {}
  std::string format() const;
};
