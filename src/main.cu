#include "lib/player.h"
#include "lib/primitive_value.h"
#include <iostream>

#define N 3

int main() {
  std::cout << "Hello, World!" << std::endl;

  const Player p1 = Player::X;
  std::cout << format_player(p1) << std::endl;

  const PrimitiveValue pv1 = PrimitiveValue::Win;
  std::cout << format_primitive_value(pv1) << std::endl;

  return 0;
}
