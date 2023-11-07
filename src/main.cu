#include "lib/board.h"
#include "lib/player.h"
#include "lib/primitive_value.h"
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;

  const Player p1 = Player::X;
  std::cout << p1.format() << std::endl;

  const PrimitiveValue pv1 = PrimitiveValue::Win;
  std::cout << pv1.format() << std::endl;

  const Board b1 = Board();
  std::cout << b1.format() << std::endl;

  return 0;
}
