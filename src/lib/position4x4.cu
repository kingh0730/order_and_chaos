#include <iostream>
#include <bitset>
#include <string>

#include "position4x4.h"
#include "position4x4_masks.h"

bool Position4x4::operator<(const Position4x4 &rhs) const
{
    int32_t l_rows = *(int32_t *)this->rows;
    int32_t r_rows = *(int32_t *)rhs.rows;

    int32_t l_rows_flipped = *(int32_t *)flip_along_x(this->rows).data();
    int32_t r_rows_flipped = *(int32_t *)flip_along_x(rhs.rows).data();

    int32_t l_cols = *(int32_t *)this->cols;
    int32_t r_cols = *(int32_t *)rhs.cols;

    int32_t l_cols_flipped = *(int32_t *)flip_along_x(this->cols).data();
    int32_t r_cols_flipped = *(int32_t *)flip_along_x(rhs.cols).data();

    return std::max({l_rows, l_cols,
                     l_rows_flipped, l_cols_flipped}) <
           std::max({r_rows, r_cols,
                     r_rows_flipped, r_cols_flipped});
}

bool Position4x4::has_4_in_a_row() const
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

std::vector<Move> Position4x4::generate_moves() const
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

Position4x4 Position4x4::do_move(const Move &move) const
{
    // Copy
    Position4x4 result = *this;

    // player
    result.player = Player(1 - result.player);

    // rows
    char_set_piece(result.rows[move.x], move.y, move.piece);

    // cols
    char_set_piece(result.cols[move.y], move.x, move.piece);

    // pos_diag
    if (move.x == move.y)
    {
        char_set_piece(result.pos_diag, move.x, move.piece);
    }

    // neg_diag
    if (move.x == 3 - move.y)
    {
        char_set_piece(result.neg_diag, move.x, move.piece);
    }

    // num_spaces_remain
    result.num_spaces_remain -= 1;

    return result;
}

PrimitiveValue Position4x4::primitive_value() const
{
    if (this->has_4_in_a_row())
    {
        switch (this->player)
        {
        case Player::Order:
            return PrimitiveValue::Win;
        case Player::Chaos:
            return PrimitiveValue::Lose;
        default:
            std::cerr << "Invalid player type: " << this->player << std::endl;
            throw std::invalid_argument("Invalid player type");
        }
    }

    // If no space remain
    if (!this->num_spaces_remain)
    {
        switch (this->player)
        {
        case Player::Order:
            return PrimitiveValue::Lose;
        case Player::Chaos:
            return PrimitiveValue::Win;
        default:
            std::cerr << "Invalid player type: " << this->player << std::endl;
            throw std::invalid_argument("Invalid player type");
        }
    }

    return PrimitiveValue::NotPrimitive;
}

// Formatting

std::string Position4x4::format() const
{
    std::string result = "";

    result += format_player(this->player) + "\t";
    result += format_primitive_value(this->primitive_value()) + '\t';
    result += "num_spaces_remain: " +
              std::to_string(this->num_spaces_remain) + '\n';

    for (int i = 0; i < 4; i++)
    {
        result += std::bitset<8>(this->rows[i]).to_string() + '\n';
    }

    return result;
}

std::string Position4x4::format_pretty() const
{
    std::string result = "";

    result += format_player(this->player) + "\t";
    result += format_primitive_value(this->primitive_value()) + '\t';
    result += "num_spaces_remain: " +
              std::to_string(this->num_spaces_remain) + '\n';

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
