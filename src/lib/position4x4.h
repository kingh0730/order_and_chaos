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

    int num_spaces_remain;

public:
    Player get_player() const { return player; }
    int get_num_spaces_remain() const { return num_spaces_remain; }

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
          num_spaces_remain(4 * 4)
    {
    }

    Position4x4(
        Player player,
        char rows[4],
        char cols[4],
        char pos_diag,
        char neg_diag,
        int num_spaces_remain)
        : player(player),
          rows{*rows},
          cols{*cols},
          pos_diag(pos_diag),
          neg_diag(neg_diag),
          num_spaces_remain(num_spaces_remain)
    {
    }

    Position4x4(const Position4x4 &other) = default;
    Position4x4 &operator=(const Position4x4 &other) = default;

    // As map key
    bool operator<(const Position4x4 &rhs) const;

    // Moves
    Position4x4 do_move(const Move &move) const;
    std::vector<Move> generate_moves() const;

    // Primitive values
    bool has_4_in_a_row() const;
    PrimitiveValue primitive_value() const;

    // Formatting
    std::string format() const;
    std::string format_pretty() const;
};
