#include "lexer.h"
#include <cctype>
#include <iostream>

Lexer::Lexer(const std::string& src)
    : source(src), position(0), line(1), column(1) {
    current_char = source.empty() ? '\0' : source[0];
}

void Lexer::advance() {
    if (current_char == '\n') {
        line++;
        column = 1;
    } else {
        column++;
    }
    position++;
    current_char = (position < source.length()) ? source[position] : '\0';
}

char Lexer::peek(int offset) {
    size_t peek_pos = position + offset;
    return (peek_pos < source.length()) ? source[peek_pos] : '\0';
}

void Lexer::skip_whitespace() {
    while (current_char != '\0' && std::isspace(current_char)) {
        advance();
    }
}

void Lexer::skip_comment() {
    if (current_char == '/' && peek() == '/') {
        while (current_char != '\0' && current_char != '\n') {
            advance();
        }
    } else if (current_char == '/' && peek() == '*') {
        advance();
        advance();
        while (current_char != '\0') {
            if (current_char == '*' && peek() == '/') {
                advance();
                advance();
                break;
            }
            advance();
        }
    }
}

Token Lexer::number() {
    int start_line = line;
    int start_col = column;
    std::string num_str;
    bool is_float = false;

    while (current_char != '\0' && (std::isdigit(current_char) || current_char == '.')) {
        if (current_char == '.') {
            if (is_float) break;
            is_float = true;
        }
        num_str += current_char;
        advance();
    }

    return Token(TokenType::TOKEN_NUMBER, num_str, start_line, start_col);
}

Token Lexer::string_literal() {
    int start_line = line;
    int start_col = column;
    std::string str;

    advance();

    while (current_char != '\0' && current_char != '"') {
        if (current_char == '\\') {
            advance();
            if (current_char == 'n') str += '\n';
            else if (current_char == 't') str += '\t';
            else if (current_char == '\\') str += '\\';
            else if (current_char == '"') str += '"';
            else str += current_char;
        } else {
            str += current_char;
        }
        advance();
    }

    if (current_char == '"') {
        advance();
    }

    return Token(TokenType::TOKEN_STRING, str, start_line, start_col);
}

Token Lexer::identifier() {
    int start_line = line;
    int start_col = column;
    std::string id_str;

    while (current_char != '\0' && (std::isalnum(current_char) || current_char == '_')) {
        id_str += current_char;
        advance();
    }

    auto it = keywords.find(id_str);
    if (it != keywords.end()) {
        return Token(it->second, id_str, start_line, start_col);
    }

    return Token(TokenType::TOKEN_IDENTIFIER, id_str, start_line, start_col);
}

Token Lexer::next_token() {
    while (current_char != '\0') {
        if (std::isspace(current_char)) {
            skip_whitespace();
            continue;
        }

        if (current_char == '/' && (peek() == '/' || peek() == '*')) {
            skip_comment();
            continue;
        }

        int start_line = line;
        int start_col = column;

        if (std::isdigit(current_char)) {
            return number();
        }

        if (std::isalpha(current_char) || current_char == '_') {
            return identifier();
        }

        if (current_char == '"') {
            return string_literal();
        }

        switch (current_char) {
            case '+':
                advance();
                return Token(TokenType::TOKEN_PLUS, "+", start_line, start_col);
            case '-':
                advance();
                return Token(TokenType::TOKEN_MINUS, "-", start_line, start_col);
            case '*':
                advance();
                return Token(TokenType::TOKEN_STAR, "*", start_line, start_col);
            case '/':
                advance();
                return Token(TokenType::TOKEN_SLASH, "/", start_line, start_col);
            case '%':
                advance();
                return Token(TokenType::TOKEN_PERCENT, "%", start_line, start_col);
            case '=':
                advance();
                if (current_char == '=') {
                    advance();
                    return Token(TokenType::TOKEN_EQ, "==", start_line, start_col);
                }
                return Token(TokenType::TOKEN_ASSIGN, "=", start_line, start_col);
            case '!':
                advance();
                if (current_char == '=') {
                    advance();
                    return Token(TokenType::TOKEN_NE, "!=", start_line, start_col);
                }
                return Token(TokenType::TOKEN_NOT, "!", start_line, start_col);
            case '<':
                advance();
                if (current_char == '=') {
                    advance();
                    return Token(TokenType::TOKEN_LE, "<=", start_line, start_col);
                }
                return Token(TokenType::TOKEN_LT, "<", start_line, start_col);
            case '>':
                advance();
                if (current_char == '=') {
                    advance();
                    return Token(TokenType::TOKEN_GE, ">=", start_line, start_col);
                }
                return Token(TokenType::TOKEN_GT, ">", start_line, start_col);
            case '&':
                advance();
                if (current_char == '&') {
                    advance();
                    return Token(TokenType::TOKEN_AND, "&&", start_line, start_col);
                }
                break;
            case '|':
                advance();
                if (current_char == '|') {
                    advance();
                    return Token(TokenType::TOKEN_OR, "||", start_line, start_col);
                }
                break;
            case '(':
                advance();
                return Token(TokenType::TOKEN_LPAREN, "(", start_line, start_col);
            case ')':
                advance();
                return Token(TokenType::TOKEN_RPAREN, ")", start_line, start_col);
            case '{':
                advance();
                return Token(TokenType::TOKEN_LBRACE, "{", start_line, start_col);
            case '}':
                advance();
                return Token(TokenType::TOKEN_RBRACE, "}", start_line, start_col);
            case '[':
                advance();
                return Token(TokenType::TOKEN_LBRACKET, "[", start_line, start_col);
            case ']':
                advance();
                return Token(TokenType::TOKEN_RBRACKET, "]", start_line, start_col);
            case ';':
                advance();
                return Token(TokenType::TOKEN_SEMICOLON, ";", start_line, start_col);
            case ',':
                advance();
                return Token(TokenType::TOKEN_COMMA, ",", start_line, start_col);
            case '.':
                advance();
                return Token(TokenType::TOKEN_DOT, ".", start_line, start_col);
            default:
                advance();
                return Token(TokenType::TOKEN_ERROR, std::string(1, current_char), start_line, start_col);
        }
    }

    return Token(TokenType::TOKEN_EOF, "", line, column);
}

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    Token token;

    do {
        token = next_token();
        tokens.push_back(token);
    } while (token.type != TokenType::TOKEN_EOF);

    return tokens;
}
