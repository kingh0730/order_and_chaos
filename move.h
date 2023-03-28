#pragma once

class Move
{
public:
    enum Piece
    {
        O = 0,
        X = 1,
    };

private:
    Piece piece;
    size_t x;
    size_t y;

public:
    Move(Piece p, size_t x, size_t y) : piece(p), x(x), y(y) {}

    void print();
};
