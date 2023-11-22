#pragma once

#include "board.h"
#include "player.h"
#include "primitive_value.h"
#include "utils.h"
#include <string>

class Position {
private:
  Player player;
  Board board;

public:
  CUDA_CALLABLE Position(Player player, Board board)
      : player(player), board(board) {}
  CUDA_CALLABLE Position(unsigned long long id, unsigned int num_empty_spaces)
      : player(num_empty_spaces), board(num_empty_spaces, id) {}
  CUDA_CALLABLE Position() : player(Player(TTT_N * TTT_N)), board() {}

  CUDA_CALLABLE unsigned long long id() const;
  CUDA_CALLABLE static unsigned long long max_id(unsigned int num_empty_spaces);
  CUDA_CALLABLE static unsigned long long
  num_positions(unsigned int num_empty_spaces);

  CUDA_CALLABLE bool is_occupied(unsigned int i, unsigned int j) const;
  CUDA_CALLABLE unsigned int children(Position *&children) const;
  CUDA_CALLABLE Position next_position(unsigned int i, unsigned int j) const;

  CUDA_CALLABLE static Board::PieceType
  player_to_piece(Player::PlayerType player);
  CUDA_CALLABLE static Player::PlayerType
  piece_to_player(Board::PieceType piece);

  CUDA_CALLABLE PrimitiveValue primitive_value() const;

  std::string format() const;
};
