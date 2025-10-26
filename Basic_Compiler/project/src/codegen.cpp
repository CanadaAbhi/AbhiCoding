#include "codegen.h"
#include <fstream>
#include <iostream>

CodeGenerator::CodeGenerator(const std::vector<IRInstruction>& instrs)
    : instructions(instrs), stack_offset(0), label_prefix(0) {}

void CodeGenerator::emit(const std::string& instruction) {
    assembly_code += "    " + instruction + "\n";
}

void CodeGenerator::emit_function_prologue(const std::string& func_name) {
    assembly_code += "\n" + func_name + ":\n";
    emit("push rbp");
    emit("mov rbp, rsp");
    emit("sub rsp, 256");
    stack_offset = 0;
    variable_offsets.clear();
}

void CodeGenerator::emit_function_epilogue() {
    emit("mov rsp, rbp");
    emit("pop rbp");
    emit("ret");
}

int CodeGenerator::get_variable_offset(const std::string& var) {
    if (variable_offsets.count(var) == 0) {
        stack_offset += 8;
        variable_offsets[var] = stack_offset;
    }
    return variable_offsets[var];
}

std::string CodeGenerator::generate() {
    assembly_code = ".section .text\n";
    assembly_code += ".global main\n";

    generate_x86_assembly();
    return assembly_code;
}

void CodeGenerator::generate_x86_assembly() {
    for (size_t i = 0; i < instructions.size(); i++) {
        const auto& inst = instructions[i];

        switch (inst.opcode) {
            case IROpcode::FUNCTION_START:
                emit_function_prologue(inst.operand1);
                break;

            case IROpcode::FUNCTION_END:
                emit_function_epilogue();
                break;

            case IROpcode::LOAD_CONST: {
                int offset = get_variable_offset(inst.result);
                emit("mov QWORD PTR [rbp-" + std::to_string(offset) + "], " + inst.operand1);
                break;
            }

            case IROpcode::LOAD: {
                int src_offset = get_variable_offset(inst.operand1);
                int dst_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(src_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(dst_offset) + "], rax");
                break;
            }

            case IROpcode::STORE: {
                int src_offset = get_variable_offset(inst.operand1);
                int dst_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(src_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(dst_offset) + "], rax");
                break;
            }

            case IROpcode::ADD: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("add rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::SUB: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("sub rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::MUL: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("imul rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::DIV: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cqo");
                emit("idiv QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::MOD: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cqo");
                emit("idiv QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rdx");
                break;
            }

            case IROpcode::NEG: {
                int operand_offset = get_variable_offset(inst.operand1);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(operand_offset) + "]");
                emit("neg rax");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::EQ: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cmp rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("sete al");
                emit("movzx rax, al");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::NE: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cmp rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("setne al");
                emit("movzx rax, al");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::LT: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cmp rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("setl al");
                emit("movzx rax, al");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::LE: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cmp rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("setle al");
                emit("movzx rax, al");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::GT: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cmp rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("setg al");
                emit("movzx rax, al");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::GE: {
                int left_offset = get_variable_offset(inst.operand1);
                int right_offset = get_variable_offset(inst.operand2);
                int result_offset = get_variable_offset(inst.result);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(left_offset) + "]");
                emit("cmp rax, QWORD PTR [rbp-" + std::to_string(right_offset) + "]");
                emit("setge al");
                emit("movzx rax, al");
                emit("mov QWORD PTR [rbp-" + std::to_string(result_offset) + "], rax");
                break;
            }

            case IROpcode::LABEL:
                assembly_code += ".L" + std::to_string(inst.label_id) + ":\n";
                break;

            case IROpcode::JUMP:
                emit("jmp .L" + std::to_string(inst.label_id));
                break;

            case IROpcode::JUMP_IF_FALSE: {
                int cond_offset = get_variable_offset(inst.operand1);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(cond_offset) + "]");
                emit("test rax, rax");
                emit("jz .L" + std::to_string(inst.label_id));
                break;
            }

            case IROpcode::JUMP_IF_TRUE: {
                int cond_offset = get_variable_offset(inst.operand1);
                emit("mov rax, QWORD PTR [rbp-" + std::to_string(cond_offset) + "]");
                emit("test rax, rax");
                emit("jnz .L" + std::to_string(inst.label_id));
                break;
            }

            case IROpcode::RETURN: {
                if (!inst.operand1.empty()) {
                    int return_offset = get_variable_offset(inst.operand1);
                    emit("mov rax, QWORD PTR [rbp-" + std::to_string(return_offset) + "]");
                } else {
                    emit("xor rax, rax");
                }
                emit_function_epilogue();
                break;
            }

            default:
                break;
        }
    }
}

void CodeGenerator::write_to_file(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << assembly_code;
        file.close();
        std::cout << "Assembly code written to " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}
