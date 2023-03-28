#include <iostream>
#include <cassert>

#include "position4x4.h"
#include "move.h"

int main()
{
    std::cout << "GamesCrafters!\n"
              << std::endl;

    Position4x4 p = Position4x4();
    std::cout << p.format();
    std::cout << p.format_pretty();

    Move m = Move(Move::Piece::X, 0, 0);
    std::cout << m.format() << '\n';

    // Generate moves
    auto moves = p.generate_moves();
    auto moves_size = moves.size();

    assert(moves_size == 32);
    std::cout << moves_size << '\n';

    for (Move &m : moves)
    {
        std::cout << m.format() << '\n';

        Position4x4 after = p.do_move(m);

        std::cout << after.format() << '\n';
        std::cout << after.format_pretty() << '\n';
    }
}
