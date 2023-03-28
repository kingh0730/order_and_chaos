#include <iostream>
#include <bitset>
#include <string>

#include "position4x4.h"

#define FOUR_OOOO 0b10101010
#define FOUR_XXXX 0b11111111

#define FOUR_TAKEN_0 0b00000010
#define FOUR_TAKEN_1 0b00001000
#define FOUR_TAKEN_2 0b00100000
#define FOUR_TAKEN_3 0b10000000

#define FOUR_CLEAR_0 0b11111100
#define FOUR_CLEAR_1 0b11110011
#define FOUR_CLEAR_2 0b11001111
#define FOUR_CLEAR_3 0b00111111

#define FOUR_SET_O_0 0b00000010
#define FOUR_SET_O_1 0b00001000
#define FOUR_SET_O_2 0b00100000
#define FOUR_SET_O_3 0b10000000

#define FOUR_SET_X_0 0b00000011
#define FOUR_SET_X_1 0b00001100
#define FOUR_SET_X_2 0b00110000
#define FOUR_SET_X_3 0b11000000

bool char_has_4_in_a_row(char c)
{
    if (c == FOUR_OOOO)
    {
        return true;
    }
    if (c == FOUR_XXXX)
    {
        return true;
    }

    return false;
}

// Position4x4

bool Position4x4::has_4_in_a_row()
{
    for (int i = 0; i < 4; i++)
    {
        if (char_has_4_in_a_row(this->rows[i]))
        {
            return true;
        }
        if (char_has_4_in_a_row(this->cols[i]))
        {
            return true;
        }
    }

    if (char_has_4_in_a_row(this->pos_diag))
    {
        return true;
    }
    if (char_has_4_in_a_row(this->neg_diag))
    {
        return true;
    }

    return false;
}

std::vector<Move> Position4x4::generate_moves()
{
    std::vector<Move> result = std::vector<Move>();

    char taken_masks[4] = {
        (char)FOUR_TAKEN_0,
        (char)FOUR_TAKEN_1,
        (char)FOUR_TAKEN_2,
        (char)FOUR_TAKEN_3};

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // If not taken
            if (!(this->rows[i] & taken_masks[j]))
            {
                result.push_back(Move(Move::Piece::O, i, j));
                result.push_back(Move(Move::Piece::X, i, j));
            }
        }
    }

    return result;
}

Position4x4 Position4x4::do_move(Move &move)
{
    // Copy
    Position4x4 result = *this;

    // TODO

    return result;
}

// Formatting

std::string Position4x4::format()
{
    std::string result = "";

    result += format_player(this->player) + '\n';

    for (int i = 0; i < 4; i++)
    {
        result += std::bitset<8>(this->rows[i]).to_string() + '\n';
    }

    return result;
}

std::string Position4x4::format_pretty()
{
    std::string result = "";

    result += format_player(this->player) + '\n';

    for (int i = 0; i < 4; i++)
    {
        // TODO
    }

    return result;
}
