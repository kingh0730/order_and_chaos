#pragma once

#include <string>

class Move
{
public:
    enum Piece
    {
        O = 0,
        X = 1,
    };

public:
    Piece piece;
    size_t x;
    size_t y;

public:
    Move(Piece p, size_t x, size_t y) : piece(p), x(x), y(y) {}

    std::string format() const;
};
