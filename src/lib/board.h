#pragma once

#include "_config.h"
#include <string>

class Board {
private:
  char board[TTT_N][TTT_N];

public:
  std::string format() const;
};
