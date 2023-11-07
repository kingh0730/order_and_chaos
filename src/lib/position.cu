#include "position.h"

std::string Position::format() const {
  auto player_str = player.format();
  auto board_str = board.format();
  return player_str + '\n' + board_str;
}
