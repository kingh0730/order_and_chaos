#pragma once

#include <vector>

#include "player.h"
#include "move.h"

class Position4x4
{
private:
    Player player;
    char rows[4];
    char cols[4];
    char pos_diag;
    char neg_diag;

public:
    Position4x4() : Position4x4(Player::Order)
    {
    }

    Position4x4(Player player)
        : player(player),
          rows{0, 0, 0, 0},
          cols{0, 0, 0, 0},
          pos_diag(0),
          neg_diag(0)
    {
    }

    Position4x4(const Position4x4 &other) = default;
    Position4x4 &operator=(const Position4x4 &other) = default;

    bool has_4_in_a_row();

    std::vector<Move> generate_moves();
    Position4x4 do_move(Move &move);

    // Formatting
    std::string format();
    std::string format_pretty();
};
