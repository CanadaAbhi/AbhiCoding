#ifndef AST_H
#define AST_H

#include <string>
#include <vector>
#include <memory>

enum class ASTNodeType {
    PROGRAM,
    FUNCTION_DECL,
    VARIABLE_DECL,
    PARAMETER,
    BLOCK,
    IF_STMT,
    WHILE_STMT,
    FOR_STMT,
    RETURN_STMT,
    BREAK_STMT,
    CONTINUE_STMT,
    EXPR_STMT,
    BINARY_EXPR,
    UNARY_EXPR,
    CALL_EXPR,
    IDENTIFIER,
    NUMBER_LITERAL,
    STRING_LITERAL,
    ASSIGNMENT
};

class ASTNode {
public:
    ASTNodeType type;
    virtual ~ASTNode() = default;
    explicit ASTNode(ASTNodeType t) : type(t) {}
};

class ExprNode : public ASTNode {
public:
    explicit ExprNode(ASTNodeType t) : ASTNode(t) {}
};

class NumberLiteral : public ExprNode {
public:
    std::string value;
    explicit NumberLiteral(const std::string& val)
        : ExprNode(ASTNodeType::NUMBER_LITERAL), value(val) {}
};

class StringLiteral : public ExprNode {
public:
    std::string value;
    explicit StringLiteral(const std::string& val)
        : ExprNode(ASTNodeType::STRING_LITERAL), value(val) {}
};

class Identifier : public ExprNode {
public:
    std::string name;
    explicit Identifier(const std::string& n)
        : ExprNode(ASTNodeType::IDENTIFIER), name(n) {}
};

class BinaryExpr : public ExprNode {
public:
    std::string op;
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;

    BinaryExpr(const std::string& operation, std::unique_ptr<ExprNode> l, std::unique_ptr<ExprNode> r)
        : ExprNode(ASTNodeType::BINARY_EXPR), op(operation), left(std::move(l)), right(std::move(r)) {}
};

class UnaryExpr : public ExprNode {
public:
    std::string op;
    std::unique_ptr<ExprNode> operand;

    UnaryExpr(const std::string& operation, std::unique_ptr<ExprNode> expr)
        : ExprNode(ASTNodeType::UNARY_EXPR), op(operation), operand(std::move(expr)) {}
};

class CallExpr : public ExprNode {
public:
    std::string function_name;
    std::vector<std::unique_ptr<ExprNode>> arguments;

    explicit CallExpr(const std::string& name)
        : ExprNode(ASTNodeType::CALL_EXPR), function_name(name) {}
};

class Assignment : public ExprNode {
public:
    std::string variable_name;
    std::unique_ptr<ExprNode> value;

    Assignment(const std::string& name, std::unique_ptr<ExprNode> val)
        : ExprNode(ASTNodeType::ASSIGNMENT), variable_name(name), value(std::move(val)) {}
};

class StmtNode : public ASTNode {
public:
    explicit StmtNode(ASTNodeType t) : ASTNode(t) {}
};

class ExprStmt : public StmtNode {
public:
    std::unique_ptr<ExprNode> expression;

    explicit ExprStmt(std::unique_ptr<ExprNode> expr)
        : StmtNode(ASTNodeType::EXPR_STMT), expression(std::move(expr)) {}
};

class Block : public StmtNode {
public:
    std::vector<std::unique_ptr<StmtNode>> statements;

    Block() : StmtNode(ASTNodeType::BLOCK) {}
};

class IfStmt : public StmtNode {
public:
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<StmtNode> then_branch;
    std::unique_ptr<StmtNode> else_branch;

    IfStmt(std::unique_ptr<ExprNode> cond, std::unique_ptr<StmtNode> then_br, std::unique_ptr<StmtNode> else_br = nullptr)
        : StmtNode(ASTNodeType::IF_STMT), condition(std::move(cond)), then_branch(std::move(then_br)), else_branch(std::move(else_br)) {}
};

class WhileStmt : public StmtNode {
public:
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<StmtNode> body;

    WhileStmt(std::unique_ptr<ExprNode> cond, std::unique_ptr<StmtNode> b)
        : StmtNode(ASTNodeType::WHILE_STMT), condition(std::move(cond)), body(std::move(b)) {}
};

class ForStmt : public StmtNode {
public:
    std::unique_ptr<StmtNode> init;
    std::unique_ptr<ExprNode> condition;
    std::unique_ptr<ExprNode> increment;
    std::unique_ptr<StmtNode> body;

    ForStmt(std::unique_ptr<StmtNode> i, std::unique_ptr<ExprNode> cond,
            std::unique_ptr<ExprNode> inc, std::unique_ptr<StmtNode> b)
        : StmtNode(ASTNodeType::FOR_STMT), init(std::move(i)), condition(std::move(cond)),
          increment(std::move(inc)), body(std::move(b)) {}
};

class ReturnStmt : public StmtNode {
public:
    std::unique_ptr<ExprNode> value;

    explicit ReturnStmt(std::unique_ptr<ExprNode> val = nullptr)
        : StmtNode(ASTNodeType::RETURN_STMT), value(std::move(val)) {}
};

class BreakStmt : public StmtNode {
public:
    BreakStmt() : StmtNode(ASTNodeType::BREAK_STMT) {}
};

class ContinueStmt : public StmtNode {
public:
    ContinueStmt() : StmtNode(ASTNodeType::CONTINUE_STMT) {}
};

class VariableDecl : public StmtNode {
public:
    std::string type_name;
    std::string var_name;
    std::unique_ptr<ExprNode> initializer;

    VariableDecl(const std::string& type, const std::string& name, std::unique_ptr<ExprNode> init = nullptr)
        : StmtNode(ASTNodeType::VARIABLE_DECL), type_name(type), var_name(name), initializer(std::move(init)) {}
};

class Parameter : public ASTNode {
public:
    std::string type_name;
    std::string param_name;

    Parameter(const std::string& type, const std::string& name)
        : ASTNode(ASTNodeType::PARAMETER), type_name(type), param_name(name) {}
};

class FunctionDecl : public ASTNode {
public:
    std::string return_type;
    std::string function_name;
    std::vector<std::unique_ptr<Parameter>> parameters;
    std::unique_ptr<Block> body;

    FunctionDecl(const std::string& ret_type, const std::string& name)
        : ASTNode(ASTNodeType::FUNCTION_DECL), return_type(ret_type), function_name(name) {}
};

class Program : public ASTNode {
public:
    std::vector<std::unique_ptr<FunctionDecl>> functions;
    std::vector<std::unique_ptr<VariableDecl>> global_variables;

    Program() : ASTNode(ASTNodeType::PROGRAM) {}
};

#endif
