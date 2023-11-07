#pragma once

#include <string>

class Player {
  enum PlayerType {
    O = 0,
    X = 1,
  };

private:
  PlayerType player;

public:
  std::string format() const;
};
