#include "board.h"
#include <iostream>

Board::Board(_PieceType b[TTT_N][TTT_N]) {
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      board[i][j] = b[i][j];
    }
  }
}

bool Board::is_win_for(_PieceType piece) const {
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

std::string Board::format(const _PieceType &piece) {
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
  return result;
}
