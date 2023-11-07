#pragma once

#include <string>

class Player {
public:
  enum PlayerType {
    O = 0,
    X = 1,
  };

private:
  PlayerType player_type;

public:
  Player(PlayerType player_type) : player_type(player_type) {}

  Player next_player() const;

  std::string format() const;
};
