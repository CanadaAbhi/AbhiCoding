#ifndef IR_H
#define IR_H

#include <string>
#include <vector>
#include <memory>

enum class IROpcode {
    NOP,
    LOAD,
    STORE,
    LOAD_CONST,
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    NEG,
    EQ,
    NE,
    LT,
    LE,
    GT,
    GE,
    AND,
    OR,
    NOT,
    JUMP,
    JUMP_IF_FALSE,
    JUMP_IF_TRUE,
    CALL,
    RETURN,
    PARAM,
    LABEL,
    FUNCTION_START,
    FUNCTION_END
};

struct IRInstruction {
    IROpcode opcode;
    std::string operand1;
    std::string operand2;
    std::string result;
    int label_id;

    IRInstruction(IROpcode op, const std::string& op1 = "", const std::string& op2 = "", const std::string& res = "")
        : opcode(op), operand1(op1), operand2(op2), result(res), label_id(-1) {}
};

class IRGenerator {
private:
    std::vector<IRInstruction> instructions;
    int temp_count;
    int label_count;

    std::string new_temp();
    int new_label();

    std::string generate_program(class Program* program);
    void generate_function(class FunctionDecl* func);
    void generate_statement(class StmtNode* stmt);
    void generate_block(class Block* block);
    std::string generate_expression(class ExprNode* expr);
    std::string generate_binary_expr(class BinaryExpr* expr);
    std::string generate_unary_expr(class UnaryExpr* expr);
    std::string generate_call_expr(class CallExpr* expr);
    std::string generate_assignment(class Assignment* assign);

public:
    IRGenerator();
    std::vector<IRInstruction> generate(class Program* program);
    void print_instructions();
};

#endif
