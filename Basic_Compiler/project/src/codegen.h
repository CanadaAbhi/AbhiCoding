#ifndef CODEGEN_H
#define CODEGEN_H

#include "ir.h"
#include <string>
#include <vector>
#include <map>

class CodeGenerator {
private:
    std::vector<IRInstruction> instructions;
    std::string assembly_code;
    std::map<std::string, int> variable_offsets;
    int stack_offset;
    int label_prefix;

    void generate_x86_assembly();
    void emit(const std::string& instruction);
    void emit_function_prologue(const std::string& func_name);
    void emit_function_epilogue();
    int get_variable_offset(const std::string& var);

public:
    explicit CodeGenerator(const std::vector<IRInstruction>& instrs);
    std::string generate();
    void write_to_file(const std::string& filename);
};

#endif
