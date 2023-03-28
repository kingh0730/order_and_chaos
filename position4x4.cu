#include <iostream>
#include <bitset>

#include "position4x4.h"

#define ROW_OOOO 0b10101010
#define ROW_XXXX 0b11111111

Position4x4::Position4x4() : Position4x4(Player::Order) {}
Position4x4::Position4x4(Player player) : player(player), rows{0, 0, 0, 0} {}
Position4x4::Position4x4(Player player, char rows[4]) : player(player), rows{*rows} {}

bool Position4x4::has_4_in_a_row()
{
    for (int i = 0; i < 4; i++)
    {
        if (this->rows[i] == ROW_OOOO)
        {
            return true;
        }
        if (this->rows[i] == ROW_XXXX)
        {
            return true;
        }
    }

    // TODO

    return false;
}

// Printing

void Position4x4::print()
{
    print_player(this->player);
    std::cout << '\n';

    for (int i = 0; i < 4; i++)
    {
        std::cout << std::bitset<8>(this->rows[i]) << '\n';
    }
}

void Position4x4::print_pretty()
{
    print_player(this->player);
    std::cout << '\n';

    for (int i = 0; i < 4; i++)
    {
        // TODO
    }
}
