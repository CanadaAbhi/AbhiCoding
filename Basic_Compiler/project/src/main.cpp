#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "lexer.h"
#include "parser.h"
#include "semantic.h"
#include "ir.h"
#include "optimizer.h"
#include "codegen.h"

std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] <input_file>\n";
    std::cout << "Options:\n";
    std::cout << "  -o <output>    Specify output file (default: output.s)\n";
    std::cout << "  -lex           Show lexical analysis (tokens)\n";
    std::cout << "  -parse         Show parse tree (AST)\n";
    std::cout << "  -ir            Show intermediate representation\n";
    std::cout << "  -opt           Show optimized IR\n";
    std::cout << "  -h, --help     Display this help message\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_file;
    std::string output_file = "output.s";
    bool show_tokens = false;
    bool show_ast = false;
    bool show_ir = false;
    bool show_optimized = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-lex") {
            show_tokens = true;
        } else if (arg == "-parse") {
            show_ast = true;
        } else if (arg == "-ir") {
            show_ir = true;
        } else if (arg == "-opt") {
            show_optimized = true;
        } else if (arg[0] != '-') {
            input_file = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (input_file.empty()) {
        std::cerr << "Error: No input file specified" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::cout << "=== Compilation Pipeline ===\n" << std::endl;

        std::string source = read_file(input_file);
        std::cout << "1. Reading source file: " << input_file << std::endl;

        std::cout << "2. Lexical Analysis (Tokenization)..." << std::endl;
        Lexer lexer(source);
        std::vector<Token> tokens = lexer.tokenize();

        if (show_tokens) {
            std::cout << "\n--- Tokens ---" << std::endl;
            for (const auto& token : tokens) {
                std::cout << "Line " << token.line << ": "
                          << static_cast<int>(token.type) << " '"
                          << token.lexeme << "'" << std::endl;
            }
        }

        std::cout << "3. Syntax Analysis (Parsing)..." << std::endl;
        Parser parser(tokens);
        auto ast = parser.parse();

        if (show_ast) {
            std::cout << "\n--- Abstract Syntax Tree ---" << std::endl;
            std::cout << "Program with " << ast->functions.size()
                      << " functions and " << ast->global_variables.size()
                      << " global variables" << std::endl;
        }

        std::cout << "4. Semantic Analysis..." << std::endl;
        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(ast.get());

        std::cout << "5. Intermediate Code Generation..." << std::endl;
        IRGenerator ir_generator;
        auto ir_instructions = ir_generator.generate(ast.get());

        if (show_ir) {
            std::cout << "\n--- Intermediate Representation ---" << std::endl;
            ir_generator.print_instructions();
        }

        std::cout << "6. Code Optimization..." << std::endl;
        Optimizer optimizer(ir_instructions);
        auto optimized_instructions = optimizer.optimize();

        if (show_optimized) {
            std::cout << "\n--- Optimized IR ---" << std::endl;
            optimizer.print_instructions();
        }

        std::cout << "7. Code Generation (x86-64 Assembly)..." << std::endl;
        CodeGenerator code_generator(optimized_instructions);
        std::string assembly = code_generator.generate();
        code_generator.write_to_file(output_file);

        std::cout << "\n=== Compilation Successful ===" << std::endl;
        std::cout << "Output written to: " << output_file << std::endl;
        std::cout << "\nTo assemble and link:" << std::endl;
        std::cout << "  as " << output_file << " -o output.o" << std::endl;
        std::cout << "  ld output.o -o program" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nCompilation Failed: " << e.what() << std::endl;
        return 1;
    }
}
