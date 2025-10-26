#include "semantic.h"
#include <iostream>
#include <stdexcept>

SymbolTable::SymbolTable() {
    enter_scope();
}

void SymbolTable::enter_scope() {
    scopes.push_back(std::map<std::string, Symbol>());
}

void SymbolTable::exit_scope() {
    if (!scopes.empty()) {
        scopes.pop_back();
    }
}

void SymbolTable::define(const Symbol& symbol) {
    if (!scopes.empty()) {
        scopes.back()[symbol.name] = symbol;
    }
}

Symbol* SymbolTable::lookup(const std::string& name) {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return &found->second;
        }
    }
    return nullptr;
}

bool SymbolTable::exists_in_current_scope(const std::string& name) {
    if (!scopes.empty()) {
        return scopes.back().find(name) != scopes.back().end();
    }
    return false;
}

SemanticAnalyzer::SemanticAnalyzer() : in_loop(false) {}

void SemanticAnalyzer::error(const std::string& message) {
    std::cerr << "Semantic Error: " << message << std::endl;
    throw std::runtime_error(message);
}

void SemanticAnalyzer::analyze(Program* program) {
    analyze_program(program);
}

void SemanticAnalyzer::analyze_program(Program* program) {
    for (auto& var : program->global_variables) {
        analyze_variable_decl(var.get());
    }

    for (auto& func : program->functions) {
        Symbol sym;
        sym.name = func->function_name;
        sym.type = func->return_type;
        sym.is_function = true;

        for (auto& param : func->parameters) {
            sym.param_types.push_back(param->type_name);
        }

        if (symbol_table.exists_in_current_scope(func->function_name)) {
            error("Function '" + func->function_name + "' already declared");
        }
        symbol_table.define(sym);
    }

    for (auto& func : program->functions) {
        analyze_function(func.get());
    }
}

void SemanticAnalyzer::analyze_function(FunctionDecl* func) {
    current_function_return_type = func->return_type;
    symbol_table.enter_scope();

    for (auto& param : func->parameters) {
        Symbol sym;
        sym.name = param->param_name;
        sym.type = param->type_name;
        sym.is_function = false;

        if (symbol_table.exists_in_current_scope(param->param_name)) {
            error("Parameter '" + param->param_name + "' already declared");
        }
        symbol_table.define(sym);
    }

    analyze_block(func->body.get());
    symbol_table.exit_scope();
}

void SemanticAnalyzer::analyze_statement(StmtNode* stmt) {
    switch (stmt->type) {
        case ASTNodeType::BLOCK:
            analyze_block(dynamic_cast<Block*>(stmt));
            break;
        case ASTNodeType::IF_STMT:
            analyze_if_stmt(dynamic_cast<IfStmt*>(stmt));
            break;
        case ASTNodeType::WHILE_STMT:
            analyze_while_stmt(dynamic_cast<WhileStmt*>(stmt));
            break;
        case ASTNodeType::FOR_STMT:
            analyze_for_stmt(dynamic_cast<ForStmt*>(stmt));
            break;
        case ASTNodeType::RETURN_STMT:
            analyze_return_stmt(dynamic_cast<ReturnStmt*>(stmt));
            break;
        case ASTNodeType::BREAK_STMT:
            if (!in_loop) {
                error("'break' statement outside loop");
            }
            break;
        case ASTNodeType::CONTINUE_STMT:
            if (!in_loop) {
                error("'continue' statement outside loop");
            }
            break;
        case ASTNodeType::EXPR_STMT: {
            auto expr_stmt = dynamic_cast<ExprStmt*>(stmt);
            analyze_expression(expr_stmt->expression.get());
            break;
        }
        case ASTNodeType::VARIABLE_DECL:
            analyze_variable_decl(dynamic_cast<VariableDecl*>(stmt));
            break;
        default:
            break;
    }
}

void SemanticAnalyzer::analyze_block(Block* block) {
    symbol_table.enter_scope();
    for (auto& stmt : block->statements) {
        analyze_statement(stmt.get());
    }
    symbol_table.exit_scope();
}

void SemanticAnalyzer::analyze_if_stmt(IfStmt* stmt) {
    std::string cond_type = analyze_expression(stmt->condition.get());
    analyze_statement(stmt->then_branch.get());
    if (stmt->else_branch) {
        analyze_statement(stmt->else_branch.get());
    }
}

void SemanticAnalyzer::analyze_while_stmt(WhileStmt* stmt) {
    std::string cond_type = analyze_expression(stmt->condition.get());
    bool prev_in_loop = in_loop;
    in_loop = true;
    analyze_statement(stmt->body.get());
    in_loop = prev_in_loop;
}

void SemanticAnalyzer::analyze_for_stmt(ForStmt* stmt) {
    symbol_table.enter_scope();

    if (stmt->init) {
        analyze_statement(stmt->init.get());
    }
    if (stmt->condition) {
        analyze_expression(stmt->condition.get());
    }
    if (stmt->increment) {
        analyze_expression(stmt->increment.get());
    }

    bool prev_in_loop = in_loop;
    in_loop = true;
    analyze_statement(stmt->body.get());
    in_loop = prev_in_loop;

    symbol_table.exit_scope();
}

void SemanticAnalyzer::analyze_return_stmt(ReturnStmt* stmt) {
    if (stmt->value) {
        std::string return_type = analyze_expression(stmt->value.get());
    } else {
        if (current_function_return_type != "void") {
            error("Non-void function must return a value");
        }
    }
}

void SemanticAnalyzer::analyze_variable_decl(VariableDecl* decl) {
    if (symbol_table.exists_in_current_scope(decl->var_name)) {
        error("Variable '" + decl->var_name + "' already declared in this scope");
    }

    Symbol sym;
    sym.name = decl->var_name;
    sym.type = decl->type_name;
    sym.is_function = false;

    if (decl->initializer) {
        std::string init_type = analyze_expression(decl->initializer.get());
    }

    symbol_table.define(sym);
}

std::string SemanticAnalyzer::analyze_expression(ExprNode* expr) {
    switch (expr->type) {
        case ASTNodeType::NUMBER_LITERAL:
            return "int";
        case ASTNodeType::STRING_LITERAL:
            return "char*";
        case ASTNodeType::IDENTIFIER: {
            auto id = dynamic_cast<Identifier*>(expr);
            Symbol* sym = symbol_table.lookup(id->name);
            if (!sym) {
                error("Undeclared identifier '" + id->name + "'");
            }
            return sym->type;
        }
        case ASTNodeType::BINARY_EXPR:
            return analyze_binary_expr(dynamic_cast<BinaryExpr*>(expr));
        case ASTNodeType::UNARY_EXPR:
            return analyze_unary_expr(dynamic_cast<UnaryExpr*>(expr));
        case ASTNodeType::CALL_EXPR:
            return analyze_call_expr(dynamic_cast<CallExpr*>(expr));
        case ASTNodeType::ASSIGNMENT:
            return analyze_assignment(dynamic_cast<Assignment*>(expr));
        default:
            error("Unknown expression type");
            return "error";
    }
}

std::string SemanticAnalyzer::analyze_binary_expr(BinaryExpr* expr) {
    std::string left_type = analyze_expression(expr->left.get());
    std::string right_type = analyze_expression(expr->right.get());

    if (expr->op == "==" || expr->op == "!=" || expr->op == "<" ||
        expr->op == "<=" || expr->op == ">" || expr->op == ">=" ||
        expr->op == "&&" || expr->op == "||") {
        return "int";
    }

    return left_type;
}

std::string SemanticAnalyzer::analyze_unary_expr(UnaryExpr* expr) {
    return analyze_expression(expr->operand.get());
}

std::string SemanticAnalyzer::analyze_call_expr(CallExpr* expr) {
    Symbol* func = symbol_table.lookup(expr->function_name);

    if (!func) {
        error("Undeclared function '" + expr->function_name + "'");
    }

    if (!func->is_function) {
        error("'" + expr->function_name + "' is not a function");
    }

    if (expr->arguments.size() != func->param_types.size()) {
        error("Function '" + expr->function_name + "' expects " +
              std::to_string(func->param_types.size()) + " arguments, got " +
              std::to_string(expr->arguments.size()));
    }

    for (auto& arg : expr->arguments) {
        analyze_expression(arg.get());
    }

    return func->type;
}

std::string SemanticAnalyzer::analyze_assignment(Assignment* assign) {
    Symbol* var = symbol_table.lookup(assign->variable_name);

    if (!var) {
        error("Undeclared variable '" + assign->variable_name + "'");
    }

    if (var->is_function) {
        error("Cannot assign to function '" + assign->variable_name + "'");
    }

    std::string value_type = analyze_expression(assign->value.get());
    return var->type;
}
