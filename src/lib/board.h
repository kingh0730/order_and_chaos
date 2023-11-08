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

private:
  static std::vector<std::map<std::string, unsigned long long>>
      id_str_to_id_for_empty_spaces;

public:
  Board(PieceType board[TTT_N][TTT_N]);

  bool is_win_for(PieceType piece) const;
  bool is_full() const;

  std::string id_str_for_empty_spaces() const;
  unsigned long long id() const;
  static unsigned long long num_boards(unsigned int num_empty_spaces);

  static std::string format(const PieceType &piece);
  std::string format() const;
};
