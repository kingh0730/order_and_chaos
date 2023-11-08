#include "board.h"
#include "utils.h"
#include <iostream>

Board::Board(PieceType b[TTT_N][TTT_N]) {
  num_empty_spaces = 0;

  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      board[i][j] = b[i][j];

      if (b[i][j] == Board::E) {
        num_empty_spaces++;
      }
    }
  }
}

bool Board::is_win_for(PieceType piece) const {
  if (piece == Board::E) {
    throw std::invalid_argument("Board::is_win_for: piece cannot be empty");
  }

  // check rows
  for (int i = 0; i < TTT_N; i++) {
    bool win = true;
    for (int j = 0; j < TTT_N; j++) {
      if (board[i][j] != piece) {
        win = false;
        break;
      }
    }
    if (win) {
      return true;
    }
  }

  // check columns
  for (int j = 0; j < TTT_N; j++) {
    bool win = true;
    for (int i = 0; i < TTT_N; i++) {
      if (board[i][j] != piece) {
        win = false;
        break;
      }
    }
    if (win) {
      return true;
    }
  }

  // check diagonals
  bool win = true;
  for (int i = 0; i < TTT_N; i++) {
    if (board[i][i] != piece) {
      win = false;
      break;
    }
  }
  if (win) {
    return true;
  }

  win = true;
  for (int i = 0; i < TTT_N; i++) {
    if (board[i][TTT_N - i - 1] != piece) {
      win = false;
      break;
    }
  }
  if (win) {
    return true;
  }

  // no win
  return false;
}

bool Board::is_full() const { return num_empty_spaces == 0; }

unsigned long long Board::id_raw_for_empty_spaces() const {
  unsigned long long result = 0;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      if (board[i][j] == Board::E) {
        result += 1 << (i * TTT_N + j);
      }
    }
  }
  return result;
}

unsigned long long Board::id() const {
  auto id = id_raw_for_empty_spaces();
  // FIXME
  return id;
}

unsigned long long Board::num_boards(unsigned int num_empty_spaces) {
  if (num_empty_spaces > TTT_N * TTT_N) {
    throw std::invalid_argument("Board::num_boards: num_empty_spaces cannot be "
                                "greater than TTT_N * TTT_N");
  }

  unsigned int num_occupied = TTT_N * TTT_N - num_empty_spaces;

  unsigned long long num_pick_empty_spaces =
      combination(TTT_N * TTT_N, num_empty_spaces);
  unsigned long long num_pick_occupied_spaces =
      1 << num_occupied; // 2^num_occupied

  return num_pick_empty_spaces * num_pick_occupied_spaces;
}

std::string Board::format(const PieceType &piece) {
  switch (piece) {
  case Board::X:
    return "X";
  case Board::O:
    return "O";
  case Board::E:
    return "_";
  default:
    std::cerr << "Board::Unknown(" + std::to_string(piece) + ")";
    throw std::invalid_argument("unknown Board piece type");
  }
}

std::string Board::format() const {
  std::string result;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      result += format(board[i][j]);
    }
    result += "\n";
  }

  result += "num_empty_spaces: " + std::to_string(num_empty_spaces) + "\n";
  result += "id: " + std::to_string(id()) + "\n";

  return result;
}
