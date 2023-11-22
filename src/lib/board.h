#pragma once

#include "_config.h"
#include "utils.h"
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
  CUDA_CALLABLE Board(const Board &other);
  CUDA_CALLABLE Board(PieceType board[TTT_N][TTT_N]);
  CUDA_CALLABLE Board(unsigned int num_empty_spaces, unsigned long long id);
  CUDA_CALLABLE Board();

  CUDA_CALLABLE unsigned int children(Board *&children, PieceType piece) const;
  CUDA_CALLABLE Board next_board(unsigned int i, unsigned int j,
                                 PieceType piece) const;

  CUDA_CALLABLE bool is_occupied(unsigned int i, unsigned int j) const;
  CUDA_CALLABLE bool is_win_for(PieceType piece) const;
  CUDA_CALLABLE bool is_full() const;
  CUDA_CALLABLE bool is_valid() const;

  CUDA_CALLABLE unsigned long long id_raw_for_empty_spaces() const;
  CUDA_CALLABLE unsigned long long id_raw_for_occupied_spaces() const;
  CUDA_CALLABLE unsigned long long id() const;
  CUDA_CALLABLE static Board board_with_max_id(unsigned int num_empty_spaces);
  CUDA_CALLABLE static unsigned long long max_id(unsigned int num_empty_spaces);
  CUDA_CALLABLE static unsigned long long
  num_boards(unsigned int num_empty_spaces);

  static std::string format(const PieceType &piece);
  std::string format() const;
};
