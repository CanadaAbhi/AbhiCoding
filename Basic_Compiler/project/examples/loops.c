int sum_to_n(int n) {
    int sum;
    int i;
    sum = 0;
    i = 1;

    while (i <= n) {
        sum = sum + i;
        i = i + 1;
    }

    return sum;
}

int main() {
    int result;
    result = sum_to_n(10);
    return result;
}
