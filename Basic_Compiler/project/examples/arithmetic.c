int calculate(int a, int b) {
    int sum;
    int diff;
    int product;
    int result;

    sum = a + b;
    diff = a - b;
    product = sum * diff;
    result = product / 2;

    return result;
}

int main() {
    int x;
    int y;
    int answer;

    x = 10;
    y = 5;
    answer = calculate(x, y);

    return answer;
}
