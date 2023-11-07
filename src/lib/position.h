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

  static Board::PieceType player_to_piece(Player::PlayerType player);
  static Player::PlayerType piece_to_player(Board::PieceType piece);

  PrimitiveValue primitive_value() const;

  std::string format() const;
};
