#include "position4x4.h"

#define ROW_OOOO 0b10101010
#define ROW_XXXX 0b11111111

bool Position4x4::has_4_in_a_row()
{
    if (this->row0 == ROW_OOOO)
    {
        return true;
    }
    if (this->row0 == ROW_XXXX)
    {
        return true;
    }
}
