#pragma once

#include <string>

class Player {
public:
  enum _PlayerType {
    O = 0,
    X = 1,
  };

private:
  _PlayerType player;

public:
  Player(_PlayerType player) : player(player) {}

  Player next_player() const;

  std::string format() const;
};
