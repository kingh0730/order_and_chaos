#include <iostream>
#include <bitset>
#include <string>

#include "position4x4.h"
#include "position4x4_masks.h"

Player Position4x4::get_player() const
{
    int parity = this->num_spaces_remain % 2;

    switch (parity)
    {
    case 0:
        return Player::Order;
    case 1:
        return Player::Chaos;
    default:
        std::cerr << "Impossible parity: " + std::to_string(parity);
        throw std::runtime_error("Impossible parity");
    }
}

uint32_t Position4x4::hash() const
{
    uint32_t rows = *(uint32_t *)this->rows;
    uint32_t cols = *(uint32_t *)this->cols;
    uint32_t rows_flip_hor = *(uint32_t *)flip_horizontal(this->rows).data();
    uint32_t cols_flip_hor = *(uint32_t *)flip_horizontal(this->cols).data();
    uint32_t rows_flip_ver = *(uint32_t *)flip_vertical(
                                  (const unsigned char(&)[4])this->rows)
                                  .data();
    uint32_t cols_flip_ver = *(uint32_t *)flip_vertical(
                                  (const unsigned char(&)[4])this->cols)
                                  .data();
    uint32_t rows_flip_ver_hor = *(uint32_t *)flip_horizontal(
                                      (const char(&)[4])rows_flip_ver)
                                      .data();
    uint32_t cols_flip_ver_hor = *(uint32_t *)flip_horizontal(
                                      (const char(&)[4])cols_flip_ver)
                                      .data();

    // Flip OX
    uint32_t rows_ox = *(uint32_t *)flip_ox((const char(&)[4])rows).data();
    uint32_t cols_ox = *(uint32_t *)flip_ox((const char(&)[4])cols).data();
    uint32_t rows_flip_hor_ox = *(uint32_t *)flip_ox(
                                     (const char(&)[4])rows_flip_hor)
                                     .data();
    uint32_t cols_flip_hor_ox = *(uint32_t *)flip_ox(
                                     (const char(&)[4])cols_flip_hor)
                                     .data();
    uint32_t rows_flip_ver_ox = *(uint32_t *)flip_ox(
                                     (const char(&)[4])rows_flip_ver)
                                     .data();
    uint32_t cols_flip_ver_ox = *(uint32_t *)flip_ox(
                                     (const char(&)[4])cols_flip_ver)
                                     .data();
    uint32_t rows_flip_ver_hor_ox = *(uint32_t *)flip_ox(
                                         (const char(&)[4])rows_flip_ver_hor)
                                         .data();
    uint32_t cols_flip_ver_hor_ox = *(uint32_t *)flip_ox(
                                         (const char(&)[4])cols_flip_ver_hor)
                                         .data();

    return std::max({
        rows,
        cols,
        rows_flip_hor,
        cols_flip_hor,
        rows_flip_ver,
        cols_flip_ver,
        rows_flip_ver_hor,
        cols_flip_ver_hor,
        rows_ox,
        cols_ox,
        rows_flip_hor_ox,
        cols_flip_hor_ox,
        rows_flip_ver_ox,
        cols_flip_ver_ox,
        rows_flip_ver_hor_ox,
        cols_flip_ver_hor_ox,
    });
}

bool Position4x4::operator<(const Position4x4 &rhs) const
{
    return this->hash() < rhs.hash();
}

bool Position4x4::has_4_in_a_row() const
{
    return int_has_4_in_a_row((const uint32_t &)this->rows);

    // for (int i = 0; i < 4; i++)
    // {
    //     if (char_has_4_in_a_row(this->rows[i]))
    //     {
    //         return true;
    //     }
    //     if (char_has_4_in_a_row(this->cols[i]))
    //     {
    //         return true;
    //     }
    // }

    // if (char_has_4_in_a_row(this->pos_diag))
    // {
    //     return true;
    // }
    // if (char_has_4_in_a_row(this->neg_diag))
    // {
    //     return true;
    // }

    // return false;
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

    // // player
    // result.player = Player(1 - result.player);

    // rows
    char_set_piece(result.rows[move.x], move.y, move.piece);

    // cols
    char_set_piece(result.cols[move.y], move.x, move.piece);

    // // pos_diag
    // if (move.x == move.y)
    // {
    //     char_set_piece(result.pos_diag, move.x, move.piece);
    // }

    // // neg_diag
    // if (move.x == 3 - move.y)
    // {
    //     char_set_piece(result.neg_diag, move.x, move.piece);
    // }

    // num_spaces_remain
    result.num_spaces_remain -= 1;

    return result;
}

PrimitiveValue Position4x4::primitive_value() const
{
    Player player = this->get_player();

    if (this->has_4_in_a_row())
    {
        switch (player)
        {
        case Player::Order:
            return PrimitiveValue::Win;
        case Player::Chaos:
            return PrimitiveValue::Lose;
        default:
            std::cerr << "Invalid player type: " << player << std::endl;
            throw std::invalid_argument("Invalid player type");
        }
    }

    // If no space remain
    if (!this->num_spaces_remain)
    {
        switch (player)
        {
        case Player::Order:
            return PrimitiveValue::Lose;
        case Player::Chaos:
            return PrimitiveValue::Win;
        default:
            std::cerr << "Invalid player type: " << player << std::endl;
            throw std::invalid_argument("Invalid player type");
        }
    }

    return PrimitiveValue::NotPrimitive;
}

// Formatting

std::string Position4x4::format() const
{
    Player player = this->get_player();

    std::string result = "";

    result += format_player(player) + "\t";
    result += format_primitive_value(this->primitive_value()) + '\t';
    result += "num_spaces_remain: " +
              std::to_string(this->num_spaces_remain) + '\n';
    result += "hash: " + std::bitset<32>(this->hash()).to_string() + '\n';

    for (int i = 0; i < 4; i++)
    {
        result += std::bitset<8>(this->rows[i]).to_string() + '\n';
    }

    return result;
}

std::string Position4x4::format_pretty() const
{
    Player player = this->get_player();

    std::string result = "";

    result += format_player(player) + "\t";
    result += format_primitive_value(this->primitive_value()) + '\t';
    result += "num_spaces_remain: " +
              std::to_string(this->num_spaces_remain) + '\n';
    result += "hash: " + std::bitset<32>(this->hash()).to_string() + '\n';

    result += "    ---------\n";
    result += "j -> 3 2 1 0\n";

    for (int i = 0; i < 4; i++)
    {
        result += 'i' + std::to_string(i) + "  |";

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

        result += '\n';
    }

    return result;
}
