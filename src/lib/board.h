#pragma once

#include "_config.h"
#include <string>

class Board {
public:
  enum _PieceType {
    O = 0,
    X = 1,
    Empty = 2,
  };

private:
  _PieceType board[TTT_N][TTT_N];

public:
  Board(_PieceType board[TTT_N][TTT_N]);

  static std::string format(const _PieceType &piece);
  std::string format() const;
};
