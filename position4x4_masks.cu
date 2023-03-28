#include <iostream>

#include "position4x4_masks.h"

bool char_has_4_in_a_row(char c)
{
    if (c == ALL_OOOO)
    {
        return true;
    }
    if (c == ALL_XXXX)
    {
        return true;
    }

    return false;
}

void char_set_piece(char &c, size_t i, Move::Piece piece)
{
    switch (piece)
    {
    case Move::Piece::O:
        c |= SET_O_MASKS[i];
        break;

    case Move::Piece::X:
        c |= SET_X_MASKS[i];
        break;

    default:
        std::cerr << "Invalid move: " << piece << std::endl;
        throw std::invalid_argument("Invalid move");
    }
}
