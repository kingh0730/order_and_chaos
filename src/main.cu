#include "lib/tier.h"
#include <chrono>
#include <iostream>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

Tier::SolveResult solve(Tier::SolveBy solve_by) {
  Tier next_tier = Tier(0, nullptr);

  auto solve_result = next_tier.solve(solve_by);
  if (solve_result != Tier::SolveResult::Success) {
    return solve_result;
  }

  for (int i = 1; i < TTT_N * TTT_N + 1; i++) {
    Tier tier = Tier(i, &next_tier);

    auto solve_result = tier.solve(solve_by);
    if (solve_result != Tier::SolveResult::Success) {
      return solve_result;
    }
  }
}

int main() {
  std::cout << "Hello, world!" << std::endl;

  auto t1 = high_resolution_clock::now();
  solve(Tier::SolveBy::CPU);
  auto t2 = high_resolution_clock::now();
  std::cout << "CPU: ";
  std::cout << duration_cast<milliseconds>(t2 - t1).count() << "ms"
            << std::endl;

  auto t3 = high_resolution_clock::now();
  solve(Tier::SolveBy::GPU);
  auto t4 = high_resolution_clock::now();
  std::cout << "GPU: ";
  std::cout << duration_cast<milliseconds>(t4 - t3).count() << "ms"
            << std::endl;

  return 0;
}
