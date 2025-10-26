#ifndef PARSER_H
#define PARSER_H

#include "token.h"
#include "ast.h"
#include <vector>
#include <memory>

class Parser {
private:
    std::vector<Token> tokens;
    size_t position;
    Token current_token;

    void advance();
    Token peek(int offset = 1);
    bool match(TokenType type);
    bool check(TokenType type);
    Token consume(TokenType type, const std::string& message);

    std::unique_ptr<Program> parse_program();
    std::unique_ptr<FunctionDecl> parse_function();
    std::unique_ptr<VariableDecl> parse_variable_decl();
    std::unique_ptr<Parameter> parse_parameter();
    std::unique_ptr<StmtNode> parse_statement();
    std::unique_ptr<Block> parse_block();
    std::unique_ptr<StmtNode> parse_if_statement();
    std::unique_ptr<StmtNode> parse_while_statement();
    std::unique_ptr<StmtNode> parse_for_statement();
    std::unique_ptr<StmtNode> parse_return_statement();
    std::unique_ptr<ExprNode> parse_expression();
    std::unique_ptr<ExprNode> parse_assignment();
    std::unique_ptr<ExprNode> parse_logical_or();
    std::unique_ptr<ExprNode> parse_logical_and();
    std::unique_ptr<ExprNode> parse_equality();
    std::unique_ptr<ExprNode> parse_relational();
    std::unique_ptr<ExprNode> parse_additive();
    std::unique_ptr<ExprNode> parse_multiplicative();
    std::unique_ptr<ExprNode> parse_unary();
    std::unique_ptr<ExprNode> parse_primary();
    std::unique_ptr<ExprNode> parse_call();

    void error(const std::string& message);

public:
    explicit Parser(const std::vector<Token>& toks);
    std::unique_ptr<Program> parse();
};

#endif
