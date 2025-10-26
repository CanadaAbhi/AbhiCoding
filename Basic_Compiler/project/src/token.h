#ifndef TOKEN_H
#define TOKEN_H

#include <string>
#include <map>

enum class TokenType {
    TOKEN_EOF,
    TOKEN_ERROR,

    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_STRING,

    TOKEN_INT,
    TOKEN_FLOAT,
    TOKEN_CHAR,
    TOKEN_VOID,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_FOR,
    TOKEN_RETURN,
    TOKEN_BREAK,
    TOKEN_CONTINUE,

    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_STAR,
    TOKEN_SLASH,
    TOKEN_PERCENT,
    TOKEN_ASSIGN,
    TOKEN_EQ,
    TOKEN_NE,
    TOKEN_LT,
    TOKEN_LE,
    TOKEN_GT,
    TOKEN_GE,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,

    TOKEN_LPAREN,
    TOKEN_RPAREN,
    TOKEN_LBRACE,
    TOKEN_RBRACE,
    TOKEN_LBRACKET,
    TOKEN_RBRACKET,
    TOKEN_SEMICOLON,
    TOKEN_COMMA,
    TOKEN_DOT
};

struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    int column;

    Token() : type(TokenType::TOKEN_EOF), lexeme(""), line(0), column(0) {}
    Token(TokenType t, const std::string& lex, int l, int c)
        : type(t), lexeme(lex), line(l), column(c) {}
};

const std::map<std::string, TokenType> keywords = {
    {"int", TokenType::TOKEN_INT},
    {"float", TokenType::TOKEN_FLOAT},
    {"char", TokenType::TOKEN_CHAR},
    {"void", TokenType::TOKEN_VOID},
    {"if", TokenType::TOKEN_IF},
    {"else", TokenType::TOKEN_ELSE},
    {"while", TokenType::TOKEN_WHILE},
    {"for", TokenType::TOKEN_FOR},
    {"return", TokenType::TOKEN_RETURN},
    {"break", TokenType::TOKEN_BREAK},
    {"continue", TokenType::TOKEN_CONTINUE}
};

#endif
