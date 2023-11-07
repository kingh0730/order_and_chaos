#include "lib/board.h"
#include "lib/player.h"
#include "lib/position.h"
#include "lib/primitive_value.h"
#include <iostream>

int main() {
  std::cout << "Hello, world!" << std::endl;

  const Player p1 = Player::X;
  std::cout << p1.format() << std::endl;

  const PrimitiveValue pv1 = PrimitiveValue::Win;
  std::cout << pv1.format() << std::endl;

  auto board1 = new Board::PieceType[TTT_N][TTT_N]{
      {Board::X, Board::O, Board::E},
      {Board::O, Board::E, Board::O},
      {Board::E, Board::O, Board::X},
  };
  const Board b1 = Board(board1);
  std::cout << b1.format() << std::endl;

  const Position position1 = Position(p1, b1);
  std::cout << position1.format() << std::endl;

  return 0;
}
