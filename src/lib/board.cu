#include "board.h"
#include "utils.h"
#include <iostream>

Board::Board(const Board &other) {
  num_empty_spaces = other.num_empty_spaces;

  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      board[i][j] = other.board[i][j];
    }
  }
}

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

bool Board::is_valid() const {
  unsigned int count_empty_spaces = 0;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      if (board[i][j] == Board::E) {
        count_empty_spaces++;
      }
    }
  }

  if (count_empty_spaces != num_empty_spaces) {
    // std::cerr << "Board::is_valid: num_empty_spaces is incorrect\n";
    return false;
  }

  return true;
}

unsigned int Board::children(Board *&children) const {
  if (!is_valid()) {
    unsigned int num_children = 0;
    children = new Board[num_children];
    return num_children;
  }

  unsigned int num_children = num_empty_spaces * 2;
  children = new Board[num_children];

  unsigned int child_idx = 0;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      if (board[i][j] == Board::E) {
        children[child_idx++] = Board(*this);
        children[child_idx - 1].board[i][j] = Board::X;
        children[child_idx++] = Board(*this);
        children[child_idx - 1].board[i][j] = Board::O;
      }
    }
  }

  return num_children;
}

Board Board::next_board(unsigned int i, unsigned int j, PieceType piece) const {
  Board next_board(*this);
  next_board.board[i][j] = piece;
  next_board.num_empty_spaces--;
  return next_board;
}

Board::Board(unsigned int num_empty_spaces, unsigned long long id)
    : num_empty_spaces(num_empty_spaces) {

  unsigned int num_occupied = TTT_N * TTT_N - num_empty_spaces;
  unsigned long long num_pick_occupied_spaces =
      1 << num_occupied; // 2^num_occupied

  auto id_raw_for_empty_spaces = id / num_pick_occupied_spaces;
  auto id_raw_for_occupied_spaces = id % num_pick_occupied_spaces;

  unsigned int num_occupied_seen = 0;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      // Take the (i * TTT_N + j) bit of id_raw_for_empty_spaces
      // and set board[i][j] to Board::E if it is 1
      if ((id_raw_for_empty_spaces >> (i * TTT_N + j)) & 1) {
        board[i][j] = Board::E;
      } else {
        // Take the num_occupied_seen bit of id_raw_for_occupied_spaces
        // and set board[i][j] to that bit
        auto bit = (id_raw_for_occupied_spaces >> num_occupied_seen++) & 1;
        board[i][j] = Board::PieceType(bit);
      }
    }
  }
}

Board::Board() {
  num_empty_spaces = TTT_N * TTT_N;

  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      board[i][j] = Board::E;
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

unsigned long long Board::id_raw_for_occupied_spaces() const {
  unsigned long long result = 0;
  unsigned int num_occupied_seen = 0;
  for (int i = 0; i < TTT_N; i++) {
    for (int j = 0; j < TTT_N; j++) {
      if (board[i][j] != Board::E) {
        result += board[i][j] << num_occupied_seen++;
      }
    }
  }
  return result;
}

unsigned long long Board::id() const {
  unsigned int num_occupied = TTT_N * TTT_N - num_empty_spaces;
  unsigned long long num_pick_occupied_spaces =
      1 << num_occupied; // 2^num_occupied

  auto id = id_raw_for_empty_spaces() * num_pick_occupied_spaces +
            id_raw_for_occupied_spaces();
  return id;
}

Board Board::board_with_max_id(unsigned int num_empty_spaces) {

  Board b_max = Board();
  b_max.num_empty_spaces = num_empty_spaces;

  unsigned int num_occupied = TTT_N * TTT_N - num_empty_spaces;

  for (int i = 0; i < num_occupied; i++) {
    b_max.board[i / TTT_N][i % TTT_N] = Board::X;
  }

  for (int i = num_occupied; i < TTT_N * TTT_N; i++) {
    b_max.board[i / TTT_N][i % TTT_N] = Board::E;
  }

  return b_max;
}

unsigned long long Board::max_id(unsigned int num_empty_spaces) {
  return board_with_max_id(num_empty_spaces).id();
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

  result += "-----";
  for (int j = 0; j < TTT_N; j++) {
    result += "--";
  }
  result += '\n';

  result += "j ->";
  for (int j = 0; j < TTT_N; j++) {
    result += ' ' + std::to_string(j);
  }
  result += '\n';

  for (int i = 0; i < TTT_N; i++) {
    result += 'i' + std::to_string(i) + "  |";
    for (int j = 0; j < TTT_N; j++) {
      result += format(board[i][j]);
      result += '|';
    }
    result += '\n';
  }

  // for (int i = 0; i < TTT_N; i++) {
  //   for (int j = 0; j < TTT_N; j++) {
  //     result += format(board[i][j]);
  //   }
  //   result += "\n";
  // }

  result += "num_empty_spaces: " + std::to_string(num_empty_spaces) + "\n";
  result += "id: " + std::to_string(id()) + "\n";
  result += "id_raw_for_empty_spaces: ";
  result += std::to_string(id_raw_for_empty_spaces()) + "\n";
  result += "id_raw_for_occupied_spaces: ";
  result += std::to_string(id_raw_for_occupied_spaces()) + "\n";

  return result;
}
