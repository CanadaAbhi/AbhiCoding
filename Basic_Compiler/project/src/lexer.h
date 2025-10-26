#ifndef LEXER_H
#define LEXER_H

#include "token.h"
#include <string>
#include <vector>

class Lexer {
private:
    std::string source;
    size_t position;
    size_t line;
    size_t column;
    char current_char;

    void advance();
    void skip_whitespace();
    void skip_comment();
    char peek(int offset = 1);
    Token number();
    Token string_literal();
    Token identifier();

public:
    Lexer(const std::string& src);
    Token next_token();
    std::vector<Token> tokenize();
};

#endif
