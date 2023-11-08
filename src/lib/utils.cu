#include "utils.h"

unsigned long long factorial(unsigned int n) {
  unsigned long long result = 1;
  for (unsigned int i = 1; i <= n; i++) {
    result *= i;
  }
  return result;
}

unsigned long long combination(unsigned int n, unsigned int k) {
  return factorial(n) / (factorial(k) * factorial(n - k));
}
