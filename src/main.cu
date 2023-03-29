#include <iostream>

#include "position4x4.h"
#include "solver.h"

int main()
{
    std::cout << "GamesCrafters!\n"
              << std::endl;

    Position4x4 p = Position4x4();
    Solver solver = Solver();
    // GameResult gr = solver.solve(p);

    // std::cout << format_game_result(gr) << '\n';

    while (true)
    {
        std::cout << p.format_pretty() << '\n';

        int input_i, input_j;
        std::cout << "Input i: ";
        std::cin >> input_i;
        std::cout << "Input j: ";
        std::cin >> input_j;

        break;
    }
}
