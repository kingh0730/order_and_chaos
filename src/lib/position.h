#pragma once

#include "board.h"
#include "player.h"
#include "primitive_value.h"
#include <string>

class Position {
private:
  Player player;
  Board board;

public:
  Position(Player player, Board board) : player(player), board(board) {}

  PrimitiveValue primitive_value() const;

  std::string format() const;
};
