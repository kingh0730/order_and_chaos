#include "position4x4_masks.h"

bool char_has_4_in_a_row(char c)
{
    if (c == ALL_OOOO)
    {
        return true;
    }
    if (c == ALL_XXXX)
    {
        return true;
    }

    return false;
}
