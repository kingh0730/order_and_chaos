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

unsigned char char_flip_vertical(unsigned char c)
{
    return c << 6 |
           (c & (unsigned char)0b00001100) << 2 |
           (c & (unsigned char)0b00110000) >> 2 |
           (c & (unsigned char)0b11000000) >> 6;
}

std::array<char, 4> flip_horizontal(const char (&chars)[4])
{
    std::array<char, 4> result = {
        chars[3],
        chars[2],
        chars[1],
        chars[0]};

    return result;
}

std::array<unsigned char, 4> flip_vertical(const unsigned char (&chars)[4])
{
    std::array<unsigned char, 4> result = {
        char_flip_vertical(chars[0]),
        char_flip_vertical(chars[1]),
        char_flip_vertical(chars[2]),
        char_flip_vertical(chars[3])};

    return result;
}

std::array<unsigned char, 4> flip_hor_ver(const unsigned char (&chars)[4])
{
    std::array<unsigned char, 4> result = {
        char_flip_vertical(chars[3]),
        char_flip_vertical(chars[2]),
        char_flip_vertical(chars[1]),
        char_flip_vertical(chars[0])};

    return result;
}

std::array<char, 4> flip_ox(const char (&chars)[4])
{
    std::array<char, 4> result = {
        char_flip_ox(chars[0]),
        char_flip_ox(chars[1]),
        char_flip_ox(chars[2]),
        char_flip_ox(chars[3])};

    return result;
}

// 32-bit

bool int_has_4_in_a_row(const uint32_t &chars)
{
    uint32_t fir_row = (chars & FIR_ROW_MASK);
    uint32_t sec_row = (chars & SEC_ROW_MASK);
    uint32_t thr_row = (chars & THR_ROW_MASK);
    uint32_t fou_row = (chars & FOU_ROW_MASK);

    uint32_t fir_col = (chars & FIR_COL_MASK);
    uint32_t sec_col = (chars & SEC_COL_MASK);
    uint32_t thr_col = (chars & THR_COL_MASK);
    uint32_t fou_col = (chars & FOU_COL_MASK);

    uint32_t pos_dia = (chars & POS_DIA_MASK);
    uint32_t neg_dia = (chars & NEG_DIA_MASK);

    return (
        fir_row == FIR_ROW_OOOO || fir_row == FIR_ROW_XXXX ||
        sec_row == SEC_ROW_OOOO || sec_row == SEC_ROW_XXXX ||
        thr_row == THR_ROW_OOOO || thr_row == THR_ROW_XXXX ||
        fou_row == FOU_ROW_OOOO || fou_row == FOU_ROW_XXXX ||
        fir_col == FIR_COL_OOOO || fir_col == FIR_COL_XXXX ||
        sec_col == SEC_COL_OOOO || sec_col == SEC_COL_XXXX ||
        thr_col == THR_COL_OOOO || thr_col == THR_COL_XXXX ||
        fou_col == FOU_COL_OOOO || fou_col == FOU_COL_XXXX ||
        pos_dia == POS_DIA_OOOO || pos_dia == POS_DIA_XXXX ||
        neg_dia == NEG_DIA_OOOO || neg_dia == NEG_DIA_XXXX);
}
