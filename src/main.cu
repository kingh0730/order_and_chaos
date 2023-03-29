#include <chrono>
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

void one_game(Solver &solver)
{
    Position4x4 p = Position4x4();

    while (true)
    {
        std::cout << p.format_pretty() << '\n';

        // Solver
        for (const Move &m : p.generate_moves())
        {
            Position4x4 after = p.do_move(m);
            GameResult gr = solver.solve(after);

            std::cout << m.format() << " -> "
                      << format_player(after.get_player()) << " "
                      << format_game_result(gr) << '\n';
        }

        std::cout << std::endl;

        // Primitive value
        PrimitiveValue pv = p.primitive_value();
        if (pv != PrimitiveValue::NotPrimitive)
        {
            std::cout << "Game over!" << '\n';
            std::cout << format_player(p.get_player()) << " ";
            std::cout << format_primitive_value(pv) << '\n'
                      << std::endl;

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
    std::cout << "GamesCrafters!\n";

    // Solve
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();

    Position4x4 p = Position4x4();
    Solver solver = Solver();
    GameResult gr = solver.solve(p);

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
    std::cout << std::endl;

    // Game loop
    while (true)
    {
        one_game(solver);
    }
}
