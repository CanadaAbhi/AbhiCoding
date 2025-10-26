#include "ir.h"
#include "ast.h"
#include <iostream>

IRGenerator::IRGenerator() : temp_count(0), label_count(0) {}

std::string IRGenerator::new_temp() {
    return "t" + std::to_string(temp_count++);
}

int IRGenerator::new_label() {
    return label_count++;
}

std::vector<IRInstruction> IRGenerator::generate(Program* program) {
    generate_program(program);
    return instructions;
}

std::string IRGenerator::generate_program(Program* program) {
    for (auto& func : program->functions) {
        generate_function(func.get());
    }
    return "";
}

void IRGenerator::generate_function(FunctionDecl* func) {
    IRInstruction func_start(IROpcode::FUNCTION_START, func->function_name);
    instructions.push_back(func_start);

    for (auto& param : func->parameters) {
        IRInstruction param_inst(IROpcode::PARAM, param->param_name);
        instructions.push_back(param_inst);
    }

    generate_block(func->body.get());

    IRInstruction func_end(IROpcode::FUNCTION_END, func->function_name);
    instructions.push_back(func_end);
}

void IRGenerator::generate_statement(StmtNode* stmt) {
    switch (stmt->type) {
        case ASTNodeType::BLOCK:
            generate_block(dynamic_cast<Block*>(stmt));
            break;

        case ASTNodeType::IF_STMT: {
            auto if_stmt = dynamic_cast<IfStmt*>(stmt);
            std::string cond = generate_expression(if_stmt->condition.get());

            int else_label = new_label();
            int end_label = new_label();

            IRInstruction jump_if_false(IROpcode::JUMP_IF_FALSE, cond);
            jump_if_false.label_id = else_label;
            instructions.push_back(jump_if_false);

            generate_statement(if_stmt->then_branch.get());

            IRInstruction jump_to_end(IROpcode::JUMP);
            jump_to_end.label_id = end_label;
            instructions.push_back(jump_to_end);

            IRInstruction else_label_inst(IROpcode::LABEL);
            else_label_inst.label_id = else_label;
            instructions.push_back(else_label_inst);

            if (if_stmt->else_branch) {
                generate_statement(if_stmt->else_branch.get());
            }

            IRInstruction end_label_inst(IROpcode::LABEL);
            end_label_inst.label_id = end_label;
            instructions.push_back(end_label_inst);
            break;
        }

        case ASTNodeType::WHILE_STMT: {
            auto while_stmt = dynamic_cast<WhileStmt*>(stmt);

            int start_label = new_label();
            int end_label = new_label();

            IRInstruction start_label_inst(IROpcode::LABEL);
            start_label_inst.label_id = start_label;
            instructions.push_back(start_label_inst);

            std::string cond = generate_expression(while_stmt->condition.get());

            IRInstruction jump_if_false(IROpcode::JUMP_IF_FALSE, cond);
            jump_if_false.label_id = end_label;
            instructions.push_back(jump_if_false);

            generate_statement(while_stmt->body.get());

            IRInstruction jump_to_start(IROpcode::JUMP);
            jump_to_start.label_id = start_label;
            instructions.push_back(jump_to_start);

            IRInstruction end_label_inst(IROpcode::LABEL);
            end_label_inst.label_id = end_label;
            instructions.push_back(end_label_inst);
            break;
        }

        case ASTNodeType::FOR_STMT: {
            auto for_stmt = dynamic_cast<ForStmt*>(stmt);

            if (for_stmt->init) {
                generate_statement(for_stmt->init.get());
            }

            int start_label = new_label();
            int end_label = new_label();
            int continue_label = new_label();

            IRInstruction start_label_inst(IROpcode::LABEL);
            start_label_inst.label_id = start_label;
            instructions.push_back(start_label_inst);

            if (for_stmt->condition) {
                std::string cond = generate_expression(for_stmt->condition.get());
                IRInstruction jump_if_false(IROpcode::JUMP_IF_FALSE, cond);
                jump_if_false.label_id = end_label;
                instructions.push_back(jump_if_false);
            }

            generate_statement(for_stmt->body.get());

            IRInstruction continue_label_inst(IROpcode::LABEL);
            continue_label_inst.label_id = continue_label;
            instructions.push_back(continue_label_inst);

            if (for_stmt->increment) {
                generate_expression(for_stmt->increment.get());
            }

            IRInstruction jump_to_start(IROpcode::JUMP);
            jump_to_start.label_id = start_label;
            instructions.push_back(jump_to_start);

            IRInstruction end_label_inst(IROpcode::LABEL);
            end_label_inst.label_id = end_label;
            instructions.push_back(end_label_inst);
            break;
        }

        case ASTNodeType::RETURN_STMT: {
            auto return_stmt = dynamic_cast<ReturnStmt*>(stmt);
            if (return_stmt->value) {
                std::string result = generate_expression(return_stmt->value.get());
                IRInstruction ret(IROpcode::RETURN, result);
                instructions.push_back(ret);
            } else {
                IRInstruction ret(IROpcode::RETURN);
                instructions.push_back(ret);
            }
            break;
        }

        case ASTNodeType::EXPR_STMT: {
            auto expr_stmt = dynamic_cast<ExprStmt*>(stmt);
            generate_expression(expr_stmt->expression.get());
            break;
        }

        case ASTNodeType::VARIABLE_DECL: {
            auto var_decl = dynamic_cast<VariableDecl*>(stmt);
            if (var_decl->initializer) {
                std::string value = generate_expression(var_decl->initializer.get());
                IRInstruction store(IROpcode::STORE, value, "", var_decl->var_name);
                instructions.push_back(store);
            }
            break;
        }

        default:
            break;
    }
}

void IRGenerator::generate_block(Block* block) {
    for (auto& stmt : block->statements) {
        generate_statement(stmt.get());
    }
}

std::string IRGenerator::generate_expression(ExprNode* expr) {
    switch (expr->type) {
        case ASTNodeType::NUMBER_LITERAL: {
            auto num = dynamic_cast<NumberLiteral*>(expr);
            std::string temp = new_temp();
            IRInstruction load_const(IROpcode::LOAD_CONST, num->value, "", temp);
            instructions.push_back(load_const);
            return temp;
        }

        case ASTNodeType::STRING_LITERAL: {
            auto str = dynamic_cast<StringLiteral*>(expr);
            std::string temp = new_temp();
            IRInstruction load_const(IROpcode::LOAD_CONST, "\"" + str->value + "\"", "", temp);
            instructions.push_back(load_const);
            return temp;
        }

        case ASTNodeType::IDENTIFIER: {
            auto id = dynamic_cast<Identifier*>(expr);
            std::string temp = new_temp();
            IRInstruction load(IROpcode::LOAD, id->name, "", temp);
            instructions.push_back(load);
            return temp;
        }

        case ASTNodeType::BINARY_EXPR:
            return generate_binary_expr(dynamic_cast<BinaryExpr*>(expr));

        case ASTNodeType::UNARY_EXPR:
            return generate_unary_expr(dynamic_cast<UnaryExpr*>(expr));

        case ASTNodeType::CALL_EXPR:
            return generate_call_expr(dynamic_cast<CallExpr*>(expr));

        case ASTNodeType::ASSIGNMENT:
            return generate_assignment(dynamic_cast<Assignment*>(expr));

        default:
            return "";
    }
}

std::string IRGenerator::generate_binary_expr(BinaryExpr* expr) {
    std::string left = generate_expression(expr->left.get());
    std::string right = generate_expression(expr->right.get());
    std::string result = new_temp();

    IROpcode opcode;
    if (expr->op == "+") opcode = IROpcode::ADD;
    else if (expr->op == "-") opcode = IROpcode::SUB;
    else if (expr->op == "*") opcode = IROpcode::MUL;
    else if (expr->op == "/") opcode = IROpcode::DIV;
    else if (expr->op == "%") opcode = IROpcode::MOD;
    else if (expr->op == "==") opcode = IROpcode::EQ;
    else if (expr->op == "!=") opcode = IROpcode::NE;
    else if (expr->op == "<") opcode = IROpcode::LT;
    else if (expr->op == "<=") opcode = IROpcode::LE;
    else if (expr->op == ">") opcode = IROpcode::GT;
    else if (expr->op == ">=") opcode = IROpcode::GE;
    else if (expr->op == "&&") opcode = IROpcode::AND;
    else if (expr->op == "||") opcode = IROpcode::OR;
    else opcode = IROpcode::NOP;

    IRInstruction inst(opcode, left, right, result);
    instructions.push_back(inst);
    return result;
}

std::string IRGenerator::generate_unary_expr(UnaryExpr* expr) {
    std::string operand = generate_expression(expr->operand.get());
    std::string result = new_temp();

    IROpcode opcode;
    if (expr->op == "-") opcode = IROpcode::NEG;
    else if (expr->op == "!") opcode = IROpcode::NOT;
    else opcode = IROpcode::NOP;

    IRInstruction inst(opcode, operand, "", result);
    instructions.push_back(inst);
    return result;
}

std::string IRGenerator::generate_call_expr(CallExpr* expr) {
    for (auto& arg : expr->arguments) {
        std::string arg_temp = generate_expression(arg.get());
        IRInstruction param(IROpcode::PARAM, arg_temp);
        instructions.push_back(param);
    }

    std::string result = new_temp();
    IRInstruction call(IROpcode::CALL, expr->function_name, std::to_string(expr->arguments.size()), result);
    instructions.push_back(call);
    return result;
}

std::string IRGenerator::generate_assignment(Assignment* assign) {
    std::string value = generate_expression(assign->value.get());
    IRInstruction store(IROpcode::STORE, value, "", assign->variable_name);
    instructions.push_back(store);
    return value;
}

void IRGenerator::print_instructions() {
    const char* opcode_names[] = {
        "NOP", "LOAD", "STORE", "LOAD_CONST", "ADD", "SUB", "MUL", "DIV", "MOD",
        "NEG", "EQ", "NE", "LT", "LE", "GT", "GE", "AND", "OR", "NOT",
        "JUMP", "JUMP_IF_FALSE", "JUMP_IF_TRUE", "CALL", "RETURN", "PARAM",
        "LABEL", "FUNCTION_START", "FUNCTION_END"
    };

    for (const auto& inst : instructions) {
        std::cout << opcode_names[static_cast<int>(inst.opcode)];

        if (!inst.operand1.empty()) std::cout << " " << inst.operand1;
        if (!inst.operand2.empty()) std::cout << ", " << inst.operand2;
        if (!inst.result.empty()) std::cout << " -> " << inst.result;
        if (inst.label_id >= 0) std::cout << " L" << inst.label_id;

        std::cout << std::endl;
    }
}
