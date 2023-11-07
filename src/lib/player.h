#pragma once

#include <string>

class Player {
public:
  enum PlayerType {
    O = 0,
    X = 1,
  };

private:
  PlayerType player;

public:
  Player(PlayerType player) : player(player) {}
  std::string format() const;
};
