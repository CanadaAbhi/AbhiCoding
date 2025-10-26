.section .text
.global main

sum_to_n:
    push rbp
    mov rbp, rsp
    sub rsp, 256
    mov QWORD PTR [rbp-8], 0
    mov rax, QWORD PTR [rbp-8]
    mov QWORD PTR [rbp-16], rax
    mov QWORD PTR [rbp-24], 1
    mov rax, QWORD PTR [rbp-24]
    mov QWORD PTR [rbp-32], rax
.L0:
    mov rax, QWORD PTR [rbp-32]
    cmp rax, QWORD PTR [rbp-40]
    setle al
    movzx rax, al
    mov QWORD PTR [rbp-48], rax
    mov rax, QWORD PTR [rbp-48]
    test rax, rax
    jz .L1
    mov rax, QWORD PTR [rbp-16]
    add rax, QWORD PTR [rbp-32]
    mov QWORD PTR [rbp-56], rax
    mov rax, QWORD PTR [rbp-56]
    mov QWORD PTR [rbp-16], rax
    mov QWORD PTR [rbp-64], 1
    mov rax, QWORD PTR [rbp-32]
    add rax, QWORD PTR [rbp-64]
    mov QWORD PTR [rbp-72], rax
    mov rax, QWORD PTR [rbp-72]
    mov QWORD PTR [rbp-32], rax
    jmp .L0
.L1:
    mov rax, QWORD PTR [rbp-16]
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
    mov rax, QWORD PTR [rbp-16]
    mov QWORD PTR [rbp-24], rax
    mov rax, QWORD PTR [rbp-24]
    mov rsp, rbp
    pop rbp
    ret
    mov rsp, rbp
    pop rbp
    ret
