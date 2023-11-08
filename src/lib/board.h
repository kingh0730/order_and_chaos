#pragma once

#include "_config.h"
#include <string>

class Board {
public:
  enum PieceType {
    O = 0,
    X = 1,
    E = 2,
  };

private:
  PieceType board[TTT_N][TTT_N];

public:
  Board(PieceType board[TTT_N][TTT_N]);

  bool is_win_for(PieceType piece) const;
  bool is_full() const;

  static unsigned long long num_boards(unsigned int num_empty_spaces);

  static std::string format(const PieceType &piece);
  std::string format() const;
};
