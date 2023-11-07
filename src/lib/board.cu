#include "board.h"
#include <iostream>

Board::Board(_PieceType b[TTT_N][TTT_N]) {
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      board[i][j] = b[i][j];
    }
  }
}

std::string Board::format(const _PieceType &piece) {
  switch (piece) {
  case Board::X:
    return "X";
  case Board::O:
    return "O";
  case Board::Empty:
    return " ";
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
