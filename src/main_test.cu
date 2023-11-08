#include "lib/board.h"
#include "lib/player.h"
#include "lib/position.h"
#include "lib/primitive_value.h"
#include "lib/recursive_value.h"
#include "lib/tier.h"
#include <cassert>
#include <iostream>

int main() {
  std::cout << "Hello, world!" << std::endl;

  const Player p1 = Player::X;
  std::cout << p1.format() << std::endl;

  const PrimitiveValue pv1 = PrimitiveValue::Win;
  std::cout << pv1.format() << std::endl;

  // Board
  auto board1 = new Board::PieceType[TTT_N][TTT_N]{
      {Board::X, Board::O, Board::E},
      {Board::O, Board::E, Board::O},
      {Board::E, Board::O, Board::X},
  };
  const Board b1 = Board(board1);
  std::cout << b1.format() << std::endl;

  const Position position1 = Position(p1, b1);
  std::cout << position1.format() << std::endl;
  std::cout << position1.primitive_value().format() << std::endl;

  assert(position1.primitive_value() == PrimitiveValue::NotPrimitive);

  // Board tests
  assert(Position(Player::X, Board(new Board::PieceType[TTT_N][TTT_N]{
                                 {Board::X, Board::X, Board::O},
                                 {Board::O, Board::E, Board::O},
                                 {Board::E, Board::O, Board::O},
                             }))
             .primitive_value() == PrimitiveValue::Lose);
  assert(Position(Player::X, Board(new Board::PieceType[TTT_N][TTT_N]{
                                 {Board::O, Board::O, Board::O},
                                 {Board::O, Board::E, Board::X},
                                 {Board::E, Board::O, Board::X},
                             }))
             .primitive_value() == PrimitiveValue::Lose);
  assert(Position(Player::X, Board(new Board::PieceType[TTT_N][TTT_N]{
                                 {Board::X, Board::X, Board::O},
                                 {Board::O, Board::O, Board::X},
                                 {Board::O, Board::O, Board::X},
                             }))
             .primitive_value() == PrimitiveValue::Lose);
  assert(Position(Player::X, Board(new Board::PieceType[TTT_N][TTT_N]{
                                 {Board::X, Board::X, Board::O},
                                 {Board::O, Board::O, Board::X},
                                 {Board::X, Board::O, Board::O},
                             }))
             .primitive_value() == PrimitiveValue::Tie);

  return 0;
}
