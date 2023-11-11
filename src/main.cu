#include "lib/tier.h"
#include <cassert>
#include <chrono>
#include <iostream>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

Tier *solve(Tier::SolveBy solve_by, bool destroy_tiers) {
  Tier *next_tier = nullptr;

  for (int i = 0; i < TTT_N * TTT_N + 1; i++) {
    Tier *tier = new Tier(i, next_tier);

    auto solve_result = tier->solve(solve_by);
    if (solve_result != Tier::SolveResult::Success) {
      throw std::runtime_error("Failed to solve tier " + std::to_string(i));
    }

    if (destroy_tiers && (next_tier != nullptr)) {
      delete next_tier;
      tier->set_next_tier_null();
    }

    next_tier = tier;
  }

  return next_tier;
}

void play_game(Tier *top_tier) {
  Board board = Board();
  Tier *tier = top_tier;

  unsigned int row;
  unsigned int col;
  while (true) {
    std::cout << "===================="
              << "====================" << std::endl;
    auto id = board.id();
    auto rv = tier->rv(id);
    std::cout << rv.format() << std::endl;

    std::cout << board.format() << std::endl;
    std::cout << "Enter i: ";
    std::cin >> row;
    if (row >= TTT_N) {
      std::cerr << "[!] Invalid i" << std::endl;
      continue;
    }
    std::cout << "Enter j: ";
    std::cin >> col;
    if (col >= TTT_N) {
      std::cerr << "[!] Invalid j" << std::endl;
      continue;
    }
  }
}

int main() {
  std::cout << "== Start!" << std::endl;

  auto t1 = high_resolution_clock::now();
  auto tier_solved_by_cpu = solve(Tier::SolveBy::CPU, false);
  auto t2 = high_resolution_clock::now();
  std::cout << "CPU: ";
  std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
            << std::endl;

  auto t3 = high_resolution_clock::now();
  auto tier_solved_by_gpu = solve(Tier::SolveBy::GPU, false);
  auto t4 = high_resolution_clock::now();
  std::cout << "GPU: ";
  std::cout << duration_cast<milliseconds>(t4 - t3).count() << "ms"
            << std::endl;

  std::cout << "== Validating..." << std::endl;
  std::cout << "CPU: " << tier_solved_by_cpu->format() << std::endl;
  std::cout << "GPU: " << tier_solved_by_gpu->format() << std::endl;
  play_game(tier_solved_by_cpu);
  assert(*tier_solved_by_cpu == *tier_solved_by_gpu);
  std::cout << "Results by CPU and GPU are the same!" << std::endl;

  return 0;
}
