#include "board.h"

std::string Board::format() const {
  std::string result;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      result += board[i][j];
    }
    result += "\n";
  }
  return result;
}
