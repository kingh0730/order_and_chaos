#pragma once

#include <vector>

#include "player.h"
#include "move.h"
#include "primitive_value.h"

class Position4x4
{
private:
    Player player;

    char rows[4];
    char cols[4];
    char pos_diag;
    char neg_diag;

    int num_spaces_remaining;

public:
    Position4x4() : Position4x4(Player::Order)
    {
    }

    Position4x4(Player player)
        : player(player),
          rows{0, 0, 0, 0},
          cols{0, 0, 0, 0},
          pos_diag(0),
          neg_diag(0),
          num_spaces_remaining(4 * 4)
    {
    }

    Position4x4(const Position4x4 &other) = default;
    Position4x4 &operator=(const Position4x4 &other) = default;

    // Moves
    Position4x4 do_move(const Move &move);
    std::vector<Move> generate_moves();

    // Primitive values
    bool has_4_in_a_row();
    PrimitiveValue primitive_value();

    // Formatting
    std::string format();
    std::string format_pretty();
};
