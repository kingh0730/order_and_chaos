#pragma once

#include "_config.h"
#include <map>
#include <string>
#include <vector>

class Board {
public:
  enum PieceType {
    O = 0,
    X = 1,
    E = 2,
  };

private:
  unsigned int num_empty_spaces;
  PieceType board[TTT_N][TTT_N];

public:
  Board(const Board &other);
  Board(PieceType board[TTT_N][TTT_N]);
  Board(unsigned int num_empty_spaces, unsigned long long id);
  Board();

  Board *children() const;

  bool is_win_for(PieceType piece) const;
  bool is_full() const;

  unsigned long long id_raw_for_empty_spaces() const;
  unsigned long long id_raw_for_occupied_spaces() const;
  unsigned long long id() const;
  static Board board_with_max_id(unsigned int num_empty_spaces);
  static unsigned long long max_id(unsigned int num_empty_spaces);
  static unsigned long long num_boards(unsigned int num_empty_spaces);

  static std::string format(const PieceType &piece);
  std::string format() const;
};
