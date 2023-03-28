#include <iostream>

#include "position4x4.h"
#include "move.h"

int main()
{
    std::cout << "GamesCrafters!\n"
              << std::endl;

    Position4x4 p = Position4x4();
    p.print();

    Move m = Move(Move::Piece::X, 0, 0);
    m.print();
}
