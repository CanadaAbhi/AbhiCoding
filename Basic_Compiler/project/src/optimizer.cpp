#include "optimizer.h"
#include <iostream>
#include <map>
#include <set>
#include <algorithm>

Optimizer::Optimizer(const std::vector<IRInstruction>& instrs) : instructions(instrs) {}

bool Optimizer::is_constant(const std::string& operand) {
    if (operand.empty()) return false;
    return std::isdigit(operand[0]) || (operand[0] == '-' && operand.length() > 1);
}

int Optimizer::evaluate_constant(const std::string& operand) {
    try {
        return std::stoi(operand);
    } catch (...) {
        return 0;
    }
}

bool Optimizer::is_temp_variable(const std::string& var) {
    return !var.empty() && var[0] == 't' && var.length() > 1;
}

void Optimizer::constant_folding() {
    for (auto& inst : instructions) {
        if (is_constant(inst.operand1) && is_constant(inst.operand2)) {
            int val1 = evaluate_constant(inst.operand1);
            int val2 = evaluate_constant(inst.operand2);
            int result = 0;

            switch (inst.opcode) {
                case IROpcode::ADD: result = val1 + val2; break;
                case IROpcode::SUB: result = val1 - val2; break;
                case IROpcode::MUL: result = val1 * val2; break;
                case IROpcode::DIV:
                    if (val2 != 0) result = val1 / val2;
                    break;
                case IROpcode::MOD:
                    if (val2 != 0) result = val1 % val2;
                    break;
                case IROpcode::EQ: result = (val1 == val2) ? 1 : 0; break;
                case IROpcode::NE: result = (val1 != val2) ? 1 : 0; break;
                case IROpcode::LT: result = (val1 < val2) ? 1 : 0; break;
                case IROpcode::LE: result = (val1 <= val2) ? 1 : 0; break;
                case IROpcode::GT: result = (val1 > val2) ? 1 : 0; break;
                case IROpcode::GE: result = (val1 >= val2) ? 1 : 0; break;
                default: continue;
            }

            inst.opcode = IROpcode::LOAD_CONST;
            inst.operand1 = std::to_string(result);
            inst.operand2 = "";
        }
    }
}

void Optimizer::dead_code_elimination() {
    std::set<std::string> used_variables;

    for (const auto& inst : instructions) {
        if (!inst.operand1.empty() && !is_constant(inst.operand1)) {
            used_variables.insert(inst.operand1);
        }
        if (!inst.operand2.empty() && !is_constant(inst.operand2)) {
            used_variables.insert(inst.operand2);
        }
    }

    std::vector<IRInstruction> new_instructions;
    for (const auto& inst : instructions) {
        bool is_used = false;

        if (inst.opcode == IROpcode::LABEL ||
            inst.opcode == IROpcode::JUMP ||
            inst.opcode == IROpcode::JUMP_IF_FALSE ||
            inst.opcode == IROpcode::JUMP_IF_TRUE ||
            inst.opcode == IROpcode::CALL ||
            inst.opcode == IROpcode::RETURN ||
            inst.opcode == IROpcode::STORE ||
            inst.opcode == IROpcode::FUNCTION_START ||
            inst.opcode == IROpcode::FUNCTION_END ||
            inst.opcode == IROpcode::PARAM) {
            is_used = true;
        }

        if (!inst.result.empty() && used_variables.count(inst.result) > 0) {
            is_used = true;
        }

        if (is_used) {
            new_instructions.push_back(inst);
        }
    }

    instructions = new_instructions;
}

void Optimizer::common_subexpression_elimination() {
    std::map<std::string, std::string> expressions;

    for (auto& inst : instructions) {
        if (inst.opcode == IROpcode::ADD || inst.opcode == IROpcode::SUB ||
            inst.opcode == IROpcode::MUL || inst.opcode == IROpcode::DIV ||
            inst.opcode == IROpcode::MOD) {

            std::string expr_key = std::to_string(static_cast<int>(inst.opcode)) + "_" +
                                   inst.operand1 + "_" + inst.operand2;

            if (expressions.count(expr_key) > 0) {
                inst.opcode = IROpcode::LOAD;
                inst.operand1 = expressions[expr_key];
                inst.operand2 = "";
            } else {
                expressions[expr_key] = inst.result;
            }
        }

        if (inst.opcode == IROpcode::STORE || inst.opcode == IROpcode::CALL) {
            expressions.clear();
        }
    }
}

void Optimizer::copy_propagation() {
    std::map<std::string, std::string> copies;

    for (auto& inst : instructions) {
        if (!inst.operand1.empty() && copies.count(inst.operand1) > 0) {
            inst.operand1 = copies[inst.operand1];
        }
        if (!inst.operand2.empty() && copies.count(inst.operand2) > 0) {
            inst.operand2 = copies[inst.operand2];
        }

        if (inst.opcode == IROpcode::LOAD && is_temp_variable(inst.result)) {
            copies[inst.result] = inst.operand1;
        }

        if (inst.opcode == IROpcode::STORE) {
            auto it = copies.begin();
            while (it != copies.end()) {
                if (it->second == inst.result) {
                    it = copies.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

std::vector<IRInstruction> Optimizer::optimize() {
    constant_folding();
    copy_propagation();
    common_subexpression_elimination();
    dead_code_elimination();

    return instructions;
}

void Optimizer::print_instructions() {
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
