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
    Move(const Piece &p, const size_t &x, const size_t &y)
        : piece(p), x(x), y(y) {}

    std::string format() const;
};
