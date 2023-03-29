#include <iostream>

#include "position4x4.h"
#include "solver.h"

Move get_input()
{
    size_t input_i;
    size_t input_j;
    Move::Piece input_piece;
    std::string input_piece_str;

    while (true)
    {
        input_i = 0;
        input_j = 0;
        input_piece_str = "";

        std::cout << "Input move...\n";

        std::cout << "Input i: ";
        std::cin >> input_i;
        if (input_i >= 4)
        {
            continue;
        }

        std::cout << "Input j: ";
        std::cin >> input_j;
        if (input_j >= 4)
        {
            continue;
        }

        std::cout << "Input piece: ";
        std::cin >> input_piece_str;

        if (input_piece_str == "o" || input_piece_str == "O")
        {
            input_piece = Move::Piece::O;
        }
        else if (input_piece_str == "x" || input_piece_str == "X")
        {
            input_piece = Move::Piece::X;
        }
        else
        {
            continue;
        }

        break;
    }

    return Move(input_piece, input_i, input_j);
}

void one_game()
{
    Position4x4 p = Position4x4();

    while (true)
    {
        std::cout << p.format_pretty() << '\n';

        // Primitive value
        PrimitiveValue pv = p.primitive_value();
        if (pv != PrimitiveValue::NotPrimitive)
        {
            std::cout << format_primitive_value(pv) << '\n';
            break;
        }

        // Inputs
        Move input_move = get_input();
        std::cout << "Chosen move: " << input_move.format() << '\n'
                  << std::endl;

        // Do move
        p = p.do_move(input_move);
    }
}

int main()
{
    std::cout << "GamesCrafters!\n"
              << std::endl;

    Position4x4 p = Position4x4();
    Solver solver = Solver();
    // GameResult gr = solver.solve(p);

    // std::cout << format_game_result(gr) << '\n';

    // TODO game loop
    one_game();
}
