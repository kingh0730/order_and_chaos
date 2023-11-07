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

  Player next_player() const;

  std::string format() const;
};
