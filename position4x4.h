#include "player.h"

class Position4x4
{
private:
    Player player;
    char rows[4];

public:
    bool has_4_in_a_row();
};
