#pragma once

#include <array>

#include "move.h"

#define A4_OOOO (char)0b10101010
#define A4_XXXX (char)0b11111111

#define TAKEN_0 (char)0b00000010
#define TAKEN_1 (char)0b00001000
#define TAKEN_2 (char)0b00100000
#define TAKEN_3 (char)0b10000000

#define CLEAR_0 (char)0b11111100
#define CLEAR_1 (char)0b11110011
#define CLEAR_2 (char)0b11001111
#define CLEAR_3 (char)0b00111111

#define SET_O_0 (char)0b00000010
#define SET_O_1 (char)0b00001000
#define SET_O_2 (char)0b00100000
#define SET_O_3 (char)0b10000000

#define SET_X_0 (char)0b00000011
#define SET_X_1 (char)0b00001100
#define SET_X_2 (char)0b00110000
#define SET_X_3 (char)0b11000000

#define FLIP_OX (char)0b01010101

const char TAKEN_MASKS[4] = {
    (char)TAKEN_0,
    (char)TAKEN_1,
    (char)TAKEN_2,
    (char)TAKEN_3};

const char CLEAR_MASKS[4] = {
    (char)CLEAR_0,
    (char)CLEAR_1,
    (char)CLEAR_2,
    (char)CLEAR_3};

const char SET_O_MASKS[4] = {
    (char)SET_O_0,
    (char)SET_O_1,
    (char)SET_O_2,
    (char)SET_O_3};

const char SET_X_MASKS[4] = {
    (char)SET_X_0,
    (char)SET_X_1,
    (char)SET_X_2,
    (char)SET_X_3};

bool char_has_4_in_a_row(const char &c);

void char_set_piece(char &c, size_t i, Move::Piece piece);

char char_flip_ox(char &c);

std::array<char, 4> flip_along_x(const char (&chars)[4]);
