#include <iostream>

#include "position4x4_masks.h"

bool char_has_4_in_a_row(const char &c)
{
    if (c == (char)A4_OOOO)
    {
        return true;
    }
    if (c == (char)A4_XXXX)
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

char char_flip_ox(char c)
{
    for (int i = 0; i < 4; i++)
    {
        if (c & TAKEN_MASKS[i])
        {
            c ^= FL_OX_MASKS[i];
        }
    }
    return c;
}

std::array<char, 4> flip_horizontal(const char (&chars)[4])
{
    std::array<char, 4> result = {chars[3], chars[2], chars[1], chars[0]};

    return result;
}

std::array<char, 4> flip_ox(char (&chars)[4])
{
    std::array<char, 4> result = {
        char_flip_ox(chars[0]),
        char_flip_ox(chars[1]),
        char_flip_ox(chars[2]),
        char_flip_ox(chars[3])};

    return result;
}
