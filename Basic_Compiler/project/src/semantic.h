#ifndef SEMANTIC_H
#define SEMANTIC_H

#include "ast.h"
#include <map>
#include <string>
#include <vector>

struct Symbol {
    std::string name;
    std::string type;
    bool is_function;
    std::vector<std::string> param_types;
};

class SymbolTable {
private:
    std::vector<std::map<std::string, Symbol>> scopes;

public:
    SymbolTable();
    void enter_scope();
    void exit_scope();
    void define(const Symbol& symbol);
    Symbol* lookup(const std::string& name);
    bool exists_in_current_scope(const std::string& name);
};

class SemanticAnalyzer {
private:
    SymbolTable symbol_table;
    std::string current_function_return_type;
    bool in_loop;

    void analyze_program(Program* program);
    void analyze_function(FunctionDecl* func);
    void analyze_statement(StmtNode* stmt);
    void analyze_block(Block* block);
    void analyze_if_stmt(IfStmt* stmt);
    void analyze_while_stmt(WhileStmt* stmt);
    void analyze_for_stmt(ForStmt* stmt);
    void analyze_return_stmt(ReturnStmt* stmt);
    void analyze_variable_decl(VariableDecl* decl);
    std::string analyze_expression(ExprNode* expr);
    std::string analyze_binary_expr(BinaryExpr* expr);
    std::string analyze_unary_expr(UnaryExpr* expr);
    std::string analyze_call_expr(CallExpr* expr);
    std::string analyze_assignment(Assignment* assign);

    void error(const std::string& message);

public:
    SemanticAnalyzer();
    void analyze(Program* program);
};

#endif
