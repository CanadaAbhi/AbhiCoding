.section .text
.global main

calculate:
    push rbp
    mov rbp, rsp
    sub rsp, 256
    mov rax, QWORD PTR [rbp-8]
    add rax, QWORD PTR [rbp-16]
    mov QWORD PTR [rbp-24], rax
    mov rax, QWORD PTR [rbp-24]
    mov QWORD PTR [rbp-32], rax
    mov rax, QWORD PTR [rbp-8]
    sub rax, QWORD PTR [rbp-16]
    mov QWORD PTR [rbp-40], rax
    mov rax, QWORD PTR [rbp-40]
    mov QWORD PTR [rbp-48], rax
    mov rax, QWORD PTR [rbp-32]
    imul rax, QWORD PTR [rbp-48]
    mov QWORD PTR [rbp-56], rax
    mov rax, QWORD PTR [rbp-56]
    mov QWORD PTR [rbp-64], rax
    mov QWORD PTR [rbp-72], 2
    mov rax, QWORD PTR [rbp-64]
    cqo
    idiv QWORD PTR [rbp-72]
    mov QWORD PTR [rbp-80], rax
    mov rax, QWORD PTR [rbp-80]
    mov QWORD PTR [rbp-88], rax
    mov rax, QWORD PTR [rbp-88]
    mov rsp, rbp
    pop rbp
    ret
    mov rsp, rbp
    pop rbp
    ret

main:
    push rbp
    mov rbp, rsp
    sub rsp, 256
    mov QWORD PTR [rbp-8], 10
    mov rax, QWORD PTR [rbp-8]
    mov QWORD PTR [rbp-16], rax
    mov QWORD PTR [rbp-24], 5
    mov rax, QWORD PTR [rbp-24]
    mov QWORD PTR [rbp-32], rax
    mov rax, QWORD PTR [rbp-40]
    mov QWORD PTR [rbp-48], rax
    mov rax, QWORD PTR [rbp-48]
    mov rsp, rbp
    pop rbp
    ret
    mov rsp, rbp
    pop rbp
    ret
