#include "position.h"
#include <iostream>

unsigned long long Position::id() const { return board.id(); }

unsigned long long Position::num_positions(unsigned int num_empty_spaces) {
  return Board::num_boards(num_empty_spaces);
}

Player::PlayerType Position::piece_to_player(Board::PieceType piece) {
  switch (piece) {
  case Board::X:
    return Player::X;
  case Board::O:
    return Player::O;
  default:
    throw std::invalid_argument("Invalid piece type");
  }
}

Board::PieceType Position::player_to_piece(Player::PlayerType player) {
  switch (player) {
  case Player::X:
    return Board::X;
  case Player::O:
    return Board::O;
  default:
    throw std::invalid_argument("Invalid player");
  }
}

PrimitiveValue Position::primitive_value() const {
  Player::PlayerType opponent_type = player.next_player_type();
  Board::PieceType opponent_piece = player_to_piece(opponent_type);

  if (board.is_win_for(opponent_piece)) {
    return PrimitiveValue::Lose;
  } else if (board.is_full()) {
    return PrimitiveValue::Tie;
  } else {
    return PrimitiveValue::NotPrimitive;
  }
}

std::string Position::format() const {
  auto player_str = player.format();
  auto board_str = board.format();
  return player_str + '\n' + board_str;
}
