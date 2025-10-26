#include "parser.h"
#include <iostream>
#include <stdexcept>

Parser::Parser(const std::vector<Token>& toks) : tokens(toks), position(0) {
    current_token = tokens.empty() ? Token() : tokens[0];
}

void Parser::advance() {
    if (position < tokens.size() - 1) {
        position++;
        current_token = tokens[position];
    }
}

Token Parser::peek(int offset) {
    size_t peek_pos = position + offset;
    return (peek_pos < tokens.size()) ? tokens[peek_pos] : Token();
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::check(TokenType type) {
    return current_token.type == type;
}

Token Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) {
        Token token = current_token;
        advance();
        return token;
    }
    error(message);
    return current_token;
}

void Parser::error(const std::string& message) {
    std::cerr << "Parse Error at line " << current_token.line
              << ", column " << current_token.column
              << ": " << message << std::endl;
    throw std::runtime_error(message);
}

std::unique_ptr<Program> Parser::parse() {
    return parse_program();
}

std::unique_ptr<Program> Parser::parse_program() {
    auto program = std::make_unique<Program>();

    while (current_token.type != TokenType::TOKEN_EOF) {
        if (check(TokenType::TOKEN_INT) || check(TokenType::TOKEN_FLOAT) ||
            check(TokenType::TOKEN_CHAR) || check(TokenType::TOKEN_VOID)) {

            std::string type = current_token.lexeme;
            advance();
            std::string name = consume(TokenType::TOKEN_IDENTIFIER, "Expected identifier").lexeme;

            if (check(TokenType::TOKEN_LPAREN)) {
                position -= 2;
                current_token = tokens[position];
                program->functions.push_back(parse_function());
            } else {
                position -= 2;
                current_token = tokens[position];
                program->global_variables.push_back(parse_variable_decl());
            }
        } else {
            error("Expected type specifier");
        }
    }

    return program;
}

std::unique_ptr<FunctionDecl> Parser::parse_function() {
    std::string return_type = current_token.lexeme;
    advance();

    std::string name = consume(TokenType::TOKEN_IDENTIFIER, "Expected function name").lexeme;
    auto func = std::make_unique<FunctionDecl>(return_type, name);

    consume(TokenType::TOKEN_LPAREN, "Expected '(' after function name");

    if (!check(TokenType::TOKEN_RPAREN)) {
        do {
            func->parameters.push_back(parse_parameter());
        } while (match(TokenType::TOKEN_COMMA));
    }

    consume(TokenType::TOKEN_RPAREN, "Expected ')' after parameters");
    func->body = parse_block();

    return func;
}

std::unique_ptr<VariableDecl> Parser::parse_variable_decl() {
    std::string type = current_token.lexeme;
    advance();

    std::string name = consume(TokenType::TOKEN_IDENTIFIER, "Expected variable name").lexeme;
    std::unique_ptr<ExprNode> initializer = nullptr;

    if (match(TokenType::TOKEN_ASSIGN)) {
        initializer = parse_expression();
    }

    consume(TokenType::TOKEN_SEMICOLON, "Expected ';' after variable declaration");
    return std::make_unique<VariableDecl>(type, name, std::move(initializer));
}

std::unique_ptr<Parameter> Parser::parse_parameter() {
    std::string type = current_token.lexeme;
    advance();

    std::string name = consume(TokenType::TOKEN_IDENTIFIER, "Expected parameter name").lexeme;
    return std::make_unique<Parameter>(type, name);
}

std::unique_ptr<StmtNode> Parser::parse_statement() {
    if (check(TokenType::TOKEN_IF)) {
        return parse_if_statement();
    }
    if (check(TokenType::TOKEN_WHILE)) {
        return parse_while_statement();
    }
    if (check(TokenType::TOKEN_FOR)) {
        return parse_for_statement();
    }
    if (check(TokenType::TOKEN_RETURN)) {
        return parse_return_statement();
    }
    if (check(TokenType::TOKEN_BREAK)) {
        advance();
        consume(TokenType::TOKEN_SEMICOLON, "Expected ';' after break");
        return std::make_unique<BreakStmt>();
    }
    if (check(TokenType::TOKEN_CONTINUE)) {
        advance();
        consume(TokenType::TOKEN_SEMICOLON, "Expected ';' after continue");
        return std::make_unique<ContinueStmt>();
    }
    if (check(TokenType::TOKEN_LBRACE)) {
        return parse_block();
    }
    if (check(TokenType::TOKEN_INT) || check(TokenType::TOKEN_FLOAT) ||
        check(TokenType::TOKEN_CHAR) || check(TokenType::TOKEN_VOID)) {
        return parse_variable_decl();
    }

    auto expr = parse_expression();
    consume(TokenType::TOKEN_SEMICOLON, "Expected ';' after expression");
    return std::make_unique<ExprStmt>(std::move(expr));
}

std::unique_ptr<Block> Parser::parse_block() {
    consume(TokenType::TOKEN_LBRACE, "Expected '{'");
    auto block = std::make_unique<Block>();

    while (!check(TokenType::TOKEN_RBRACE) && current_token.type != TokenType::TOKEN_EOF) {
        block->statements.push_back(parse_statement());
    }

    consume(TokenType::TOKEN_RBRACE, "Expected '}'");
    return block;
}

std::unique_ptr<StmtNode> Parser::parse_if_statement() {
    consume(TokenType::TOKEN_IF, "Expected 'if'");
    consume(TokenType::TOKEN_LPAREN, "Expected '(' after 'if'");
    auto condition = parse_expression();
    consume(TokenType::TOKEN_RPAREN, "Expected ')' after condition");

    auto then_branch = parse_statement();
    std::unique_ptr<StmtNode> else_branch = nullptr;

    if (match(TokenType::TOKEN_ELSE)) {
        else_branch = parse_statement();
    }

    return std::make_unique<IfStmt>(std::move(condition), std::move(then_branch), std::move(else_branch));
}

std::unique_ptr<StmtNode> Parser::parse_while_statement() {
    consume(TokenType::TOKEN_WHILE, "Expected 'while'");
    consume(TokenType::TOKEN_LPAREN, "Expected '(' after 'while'");
    auto condition = parse_expression();
    consume(TokenType::TOKEN_RPAREN, "Expected ')' after condition");

    auto body = parse_statement();
    return std::make_unique<WhileStmt>(std::move(condition), std::move(body));
}

std::unique_ptr<StmtNode> Parser::parse_for_statement() {
    consume(TokenType::TOKEN_FOR, "Expected 'for'");
    consume(TokenType::TOKEN_LPAREN, "Expected '(' after 'for'");

    std::unique_ptr<StmtNode> init = nullptr;
    if (!check(TokenType::TOKEN_SEMICOLON)) {
        if (check(TokenType::TOKEN_INT) || check(TokenType::TOKEN_FLOAT) ||
            check(TokenType::TOKEN_CHAR)) {
            init = parse_variable_decl();
        } else {
            auto expr = parse_expression();
            consume(TokenType::TOKEN_SEMICOLON, "Expected ';'");
            init = std::make_unique<ExprStmt>(std::move(expr));
        }
    } else {
        advance();
    }

    std::unique_ptr<ExprNode> condition = nullptr;
    if (!check(TokenType::TOKEN_SEMICOLON)) {
        condition = parse_expression();
    }
    consume(TokenType::TOKEN_SEMICOLON, "Expected ';'");

    std::unique_ptr<ExprNode> increment = nullptr;
    if (!check(TokenType::TOKEN_RPAREN)) {
        increment = parse_expression();
    }
    consume(TokenType::TOKEN_RPAREN, "Expected ')' after for clauses");

    auto body = parse_statement();
    return std::make_unique<ForStmt>(std::move(init), std::move(condition), std::move(increment), std::move(body));
}

std::unique_ptr<StmtNode> Parser::parse_return_statement() {
    consume(TokenType::TOKEN_RETURN, "Expected 'return'");
    std::unique_ptr<ExprNode> value = nullptr;

    if (!check(TokenType::TOKEN_SEMICOLON)) {
        value = parse_expression();
    }

    consume(TokenType::TOKEN_SEMICOLON, "Expected ';' after return");
    return std::make_unique<ReturnStmt>(std::move(value));
}

std::unique_ptr<ExprNode> Parser::parse_expression() {
    return parse_assignment();
}

std::unique_ptr<ExprNode> Parser::parse_assignment() {
    auto expr = parse_logical_or();

    if (match(TokenType::TOKEN_ASSIGN)) {
        if (expr->type == ASTNodeType::IDENTIFIER) {
            auto id = dynamic_cast<Identifier*>(expr.get());
            auto value = parse_assignment();
            return std::make_unique<Assignment>(id->name, std::move(value));
        }
        error("Invalid assignment target");
    }

    return expr;
}

std::unique_ptr<ExprNode> Parser::parse_logical_or() {
    auto left = parse_logical_and();

    while (match(TokenType::TOKEN_OR)) {
        std::string op = "||";
        auto right = parse_logical_and();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }

    return left;
}

std::unique_ptr<ExprNode> Parser::parse_logical_and() {
    auto left = parse_equality();

    while (match(TokenType::TOKEN_AND)) {
        std::string op = "&&";
        auto right = parse_equality();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }

    return left;
}

std::unique_ptr<ExprNode> Parser::parse_equality() {
    auto left = parse_relational();

    while (check(TokenType::TOKEN_EQ) || check(TokenType::TOKEN_NE)) {
        std::string op = current_token.lexeme;
        advance();
        auto right = parse_relational();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }

    return left;
}

std::unique_ptr<ExprNode> Parser::parse_relational() {
    auto left = parse_additive();

    while (check(TokenType::TOKEN_LT) || check(TokenType::TOKEN_LE) ||
           check(TokenType::TOKEN_GT) || check(TokenType::TOKEN_GE)) {
        std::string op = current_token.lexeme;
        advance();
        auto right = parse_additive();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }

    return left;
}

std::unique_ptr<ExprNode> Parser::parse_additive() {
    auto left = parse_multiplicative();

    while (check(TokenType::TOKEN_PLUS) || check(TokenType::TOKEN_MINUS)) {
        std::string op = current_token.lexeme;
        advance();
        auto right = parse_multiplicative();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }

    return left;
}

std::unique_ptr<ExprNode> Parser::parse_multiplicative() {
    auto left = parse_unary();

    while (check(TokenType::TOKEN_STAR) || check(TokenType::TOKEN_SLASH) || check(TokenType::TOKEN_PERCENT)) {
        std::string op = current_token.lexeme;
        advance();
        auto right = parse_unary();
        left = std::make_unique<BinaryExpr>(op, std::move(left), std::move(right));
    }

    return left;
}

std::unique_ptr<ExprNode> Parser::parse_unary() {
    if (check(TokenType::TOKEN_MINUS) || check(TokenType::TOKEN_NOT)) {
        std::string op = current_token.lexeme;
        advance();
        auto operand = parse_unary();
        return std::make_unique<UnaryExpr>(op, std::move(operand));
    }

    return parse_call();
}

std::unique_ptr<ExprNode> Parser::parse_call() {
    auto expr = parse_primary();

    if (check(TokenType::TOKEN_LPAREN)) {
        if (expr->type == ASTNodeType::IDENTIFIER) {
            auto id = dynamic_cast<Identifier*>(expr.get());
            auto call = std::make_unique<CallExpr>(id->name);

            advance();

            if (!check(TokenType::TOKEN_RPAREN)) {
                do {
                    call->arguments.push_back(parse_expression());
                } while (match(TokenType::TOKEN_COMMA));
            }

            consume(TokenType::TOKEN_RPAREN, "Expected ')' after arguments");
            return call;
        }
    }

    return expr;
}

std::unique_ptr<ExprNode> Parser::parse_primary() {
    if (check(TokenType::TOKEN_NUMBER)) {
        std::string value = current_token.lexeme;
        advance();
        return std::make_unique<NumberLiteral>(value);
    }

    if (check(TokenType::TOKEN_STRING)) {
        std::string value = current_token.lexeme;
        advance();
        return std::make_unique<StringLiteral>(value);
    }

    if (check(TokenType::TOKEN_IDENTIFIER)) {
        std::string name = current_token.lexeme;
        advance();
        return std::make_unique<Identifier>(name);
    }

    if (match(TokenType::TOKEN_LPAREN)) {
        auto expr = parse_expression();
        consume(TokenType::TOKEN_RPAREN, "Expected ')' after expression");
        return expr;
    }

    error("Expected expression");
    return nullptr;
}
