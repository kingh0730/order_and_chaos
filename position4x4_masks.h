#pragma once

#include "move.h"

#define ALL_OOOO 0b10101010
#define ALL_XXXX 0b11111111

#define TAKEN_0 0b00000010
#define TAKEN_1 0b00001000
#define TAKEN_2 0b00100000
#define TAKEN_3 0b10000000

#define CLEAR_0 0b11111100
#define CLEAR_1 0b11110011
#define CLEAR_2 0b11001111
#define CLEAR_3 0b00111111

#define SET_O_0 0b00000010
#define SET_O_1 0b00001000
#define SET_O_2 0b00100000
#define SET_O_3 0b10000000

#define SET_X_0 0b00000011
#define SET_X_1 0b00001100
#define SET_X_2 0b00110000
#define SET_X_3 0b11000000

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

bool char_has_4_in_a_row(char c);

void char_set_piece(char &c, size_t i, Move::Piece piece);
