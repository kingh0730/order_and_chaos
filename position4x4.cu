#include <iostream>
#include <bitset>
#include <string>

#include "position4x4.h"
#include "position4x4_masks.h"

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

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // If not taken
            if (!(this->rows[i] & TAKEN_MASKS[j]))
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

    // player
    result.player = Player(1 - result.player);

    // rows
    result.rows[move.x] &= CLEAR_MASKS[move.y];

    switch (move.piece)
    {
    case Move::Piece::O:
        result.rows[move.x] |= SET_O_MASKS[move.y];
        break;

    case Move::Piece::X:
        result.rows[move.x] |= SET_X_MASKS[move.y];
        break;

    default:
        std::cerr << "Invalid move: " << move.piece << std::endl;
        throw std::invalid_argument("Invalid move");
    }

    // TODO

    // num_spaces_remaining
    result.num_spaces_remaining -= 1;

    return result;
}

// Formatting

std::string Position4x4::format()
{
    std::string result = "";

    result += format_player(this->player) + "\t";
    result += "num_spaces_remaining: " +
              std::to_string(this->num_spaces_remaining) + '\n';

    for (int i = 0; i < 4; i++)
    {
        result += std::bitset<8>(this->rows[i]).to_string() + '\n';
    }

    return result;
}

std::string Position4x4::format_pretty()
{
    std::string result = "";

    result += format_player(this->player) + "\t";
    result += "num_spaces_remaining: " +
              std::to_string(this->num_spaces_remaining) + '\n';

    for (int i = 0; i < 4; i++)
    {
        result += '|';

        char row = this->rows[i];

        for (int j = 0; j < 4; j++)
        {
            if (row & TAKEN_MASKS[3])
            {
                if (row >= (char)SET_X_3)
                {
                    result += 'X';
                }
                else
                {
                    result += 'O';
                }
            }
            else
            {
                result += ' ';
            }

            result += '|';

            row <<= 2;
        }

        result += "\n---------\n";
    }

    return result;
}
