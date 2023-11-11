#include "position.h"
#include <iostream>

unsigned long long Position::id() const { return board.id(); }

unsigned long long Position::max_id(unsigned int num_empty_spaces) {
  return Board::max_id(num_empty_spaces);
}

unsigned long long Position::num_positions(unsigned int num_empty_spaces) {
  return Board::num_boards(num_empty_spaces);
}

unsigned int Position::children(Position *&children) const {
  Board::PieceType next_piece = player_to_piece(player.get_player_type());
  Board *child_boards;
  unsigned int num_children = board.children(child_boards, next_piece);
  children = new Position[num_children];

  for (unsigned int i = 0; i < num_children; i++) {
    children[i] = Position(player.next_player_type(), child_boards[i]);
  }

  delete[] child_boards;
  return num_children;
}

bool Position::is_occupied(unsigned int i, unsigned int j) const {
  return board.is_occupied(i, j);
}

Position Position::next_position(unsigned int i, unsigned int j) const {
  Player::PlayerType next_player_type = player.next_player_type();
  Board next_board = board.next_board(i, j, player_to_piece(next_player_type));
  return Position(next_player_type, next_board);
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
