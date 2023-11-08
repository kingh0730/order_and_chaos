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
void test_tiers();

int main() {
  std::cout << "\n== Test utils ==" << std::endl;
  test_utils();

  std::cout << "\n== Test boards ==" << std::endl;
  test_boards();

  std::cout << "\n== Test tiers ==" << std::endl;
  test_tiers();

  return 0;
}

void test_tiers() {
  Tier tier0 = Tier(0, nullptr);
  tier0.solve(Tier::SolveBy::CPU);
  std::cout << tier0.format() << std::endl;

  Tier tier1 = Tier(1, &tier0);
  tier1.solve(Tier::SolveBy::GPU);
  std::cout << tier1.format() << std::endl;

  Tier tier2 = Tier(2, &tier1);
  tier2.solve(Tier::SolveBy::CPU);
  std::cout << tier2.format() << std::endl;

  Tier tier3 = Tier(3, &tier2);
  tier3.solve(Tier::SolveBy::GPU);
  std::cout << tier3.format() << std::endl;

  Tier tier4 = Tier(4, &tier3);
  tier4.solve(Tier::SolveBy::GPU);
  std::cout << tier4.format() << std::endl;

  Tier tier5 = Tier(5, &tier4);
  tier5.solve(Tier::SolveBy::CPU);
  std::cout << tier5.format() << std::endl;

  Tier tier6 = Tier(6, &tier5);
  tier6.solve(Tier::SolveBy::GPU);
  std::cout << tier6.format() << std::endl;

  Tier tier7 = Tier(7, &tier6);
  tier7.solve(Tier::SolveBy::CPU);
  std::cout << tier7.format() << std::endl;

  Tier tier8 = Tier(8, &tier7);
  tier8.solve(Tier::SolveBy::GPU);
  std::cout << tier8.format() << std::endl;

  Tier tier9 = Tier(9, &tier8);
  std::cout << tier9.format() << std::endl;
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

  auto board2 = new Board::PieceType[TTT_N][TTT_N]{
      {Board::E, Board::O, Board::X},
      {Board::O, Board::X, Board::O},
      {Board::X, Board::O, Board::X},
  };
  const Board b2 = Board(board2);
  std::cout << b2.format() << std::endl;

  auto board3 = new Board::PieceType[TTT_N][TTT_N]{
      {Board::X, Board::O, Board::X},
      {Board::O, Board::O, Board::O},
      {Board::X, Board::O, Board::E},
  };
  const Board b3 = Board(board3);
  std::cout << b3.format() << std::endl;

  auto board4 = new Board::PieceType[TTT_N][TTT_N]{
      {Board::X, Board::O, Board::X},
      {Board::O, Board::O, Board::E},
      {Board::X, Board::O, Board::X},
  };
  const Board b4 = Board(board4);
  std::cout << b4.format() << std::endl;

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

  // number of boards
  assert(Board::num_boards(0) == 512);
  assert(Board::num_boards(1) == 2304);

  // max_id
  std::cout << Board::board_with_max_id(0).format() << std::endl;
  std::cout << Board::board_with_max_id(1).format() << std::endl;
  std::cout << Board::board_with_max_id(2).format() << std::endl;
  std::cout << Board::board_with_max_id(7).format() << std::endl;
  std::cout << Board::board_with_max_id(8).format() << std::endl;
  std::cout << Board::board_with_max_id(9).format() << std::endl;
}
