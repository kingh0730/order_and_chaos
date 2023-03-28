#include "player.h"

class Position4x4
{
private:
    Player player;
    char row0;
    char row1;
    char row2;
    char row3;

public:
    bool has_4_in_a_row();
};
