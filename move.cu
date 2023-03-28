#include <iostream>

#include "move.h"

void Move::print()
{
    switch (this->piece)
    {
    case Piece::O:
        std::cout << "Piece::O";
        break;

    case Piece::X:
        std::cout << "Piece::X";
        break;

    default:
        std::cout << "Piece::Unknown\n";
        std::cout << "value: " << this->piece << std::endl;
        throw std::invalid_argument("unknown piece type");
    }

    printf(" (%zd, %zd)", this->x, this->y);
}
