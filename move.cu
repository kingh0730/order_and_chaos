#include <iostream>

#include "move.h"

std::string Move::format()
{
    std::string result = "";

    switch (this->piece)
    {
    case Piece::O:
        result += "Piece::O";
        break;

    case Piece::X:
        result += "Piece::X";
        break;

    default:
        std::cerr << "Piece::Unknown\n";
        std::cerr << "value: " << this->piece << std::endl;
        throw std::invalid_argument("unknown piece type");
    }

    result += " (" + std::to_string(this->x);
    result += ", " + std::to_string(this->y);
    result += ")";

    return result;
}
