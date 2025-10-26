#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "ir.h"
#include <vector>

class Optimizer {
private:
    std::vector<IRInstruction> instructions;

    void constant_folding();
    void dead_code_elimination();
    void common_subexpression_elimination();
    void copy_propagation();

    bool is_constant(const std::string& operand);
    int evaluate_constant(const std::string& operand);
    bool is_temp_variable(const std::string& var);

public:
    explicit Optimizer(const std::vector<IRInstruction>& instrs);
    std::vector<IRInstruction> optimize();
    void print_instructions();
};

#endif
