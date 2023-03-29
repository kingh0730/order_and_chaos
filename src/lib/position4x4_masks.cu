#include <iostream>

#include "position4x4_masks.h"

bool char_has_4_in_a_row(const char &c)
{
    if (c == (char)ALL_OOOO)
    {
        return true;
    }
    if (c == (char)ALL_XXXX)
    {
        return true;
    }

    return false;
}

void char_set_piece(char &c, size_t i, Move::Piece piece)
{
    // FIXME This line is perhaps not necessary.
    // c &= CLEAR_MASKS[i];

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

std::array<char, 4> flip_along_x(const char (&chars)[4])
{
    std::array<char, 4> result = {chars[3], chars[2], chars[1], chars[0]};

    return result;
}
