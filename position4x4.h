#pragma once

#include "player.h"

class Position4x4
{
private:
    Player player;
    char rows[4];

public:
    Position4x4();
    Position4x4(Player player);
    Position4x4(Player player, char rows[4]);

    Position4x4(const Position4x4 &other) = default;
    Position4x4 &operator=(const Position4x4 &other) = default;

    bool has_4_in_a_row();

    // Printing
    void print();
    void print_pretty();
};
