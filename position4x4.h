#include "player.h"

class Position4x4
{
private:
    Player player;
    char rows[4];

public:
    Position4x4(Player player);
    bool has_4_in_a_row();
};
