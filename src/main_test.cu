#include "lib/board.h"
#include "lib/player.h"
#include "lib/position.h"
#include "lib/primitive_value.h"
#include "lib/recursive_value.h"
#include "lib/tier.h"
#include "lib/utils.h"
#include <cassert>
#include <iostream>

void test_utils();
void test_boards();

int main() {
  std::cout << "== Test utils ==" << std::endl;
  test_utils();

  std::cout << "== Test boards ==" << std::endl;
  test_boards();

  // Tier
  Tier tier0 = Tier(0, nullptr);

  tier0.solve(Tier::SolveBy::CPU);
  std::cout << tier0.format() << std::endl;

  return 0;
}

void test_utils() {
  assert(factorial(0) == 1);
  assert(factorial(1) == 1);
  assert(factorial(2) == 2);
  assert(factorial(3) == 6);
  assert(factorial(4) == 24);
  assert(factorial(5) == 120);
  assert(factorial(6) == 720);
  assert(factorial(7) == 5040);
  assert(factorial(8) == 40320);
  assert(factorial(9) == 362880);

  assert(combination(0, 0) == 1);
  assert(combination(1, 0) == 1);
  assert(combination(1, 1) == 1);
  assert(combination(2, 0) == 1);
  assert(combination(2, 1) == 2);
  assert(combination(2, 2) == 1);
  assert(combination(3, 0) == 1);
  assert(combination(3, 1) == 3);
  assert(combination(3, 2) == 3);
  assert(combination(3, 3) == 1);
  assert(combination(4, 0) == 1);
  assert(combination(4, 1) == 4);
  assert(combination(4, 2) == 6);
  assert(combination(4, 3) == 4);
  assert(combination(4, 4) == 1);
  assert(combination(5, 0) == 1);
  assert(combination(5, 1) == 5);
  assert(combination(5, 2) == 10);
  assert(combination(5, 3) == 10);
  assert(combination(5, 4) == 5);
  assert(combination(5, 5) == 1);
  assert(combination(6, 0) == 1);
  assert(combination(6, 1) == 6);
  assert(combination(6, 2) == 15);
  assert(combination(6, 3) == 20);
  assert(combination(6, 4) == 15);
  assert(combination(6, 5) == 6);
  assert(combination(6, 6) == 1);
}

void test_boards() {
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

  // Boards
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
}
