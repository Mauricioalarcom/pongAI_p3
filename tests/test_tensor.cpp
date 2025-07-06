#include "utec/algebra/tensor.h"
#include <iostream>
#include <cassert>

using namespace utec::algebra;

void test_case_1() {
    std::cout << "Test Case 1: Creación, acceso y fill\n";
    Tensor<int,2> t(2,3);
    t.fill(7);
    int x = t(1,2);
    assert(x == 7);
    std::cout << "✓ x == 7: " << x << "\n\n";
}

void test_case_2() {
    std::cout << "Test Case 2: Reshape válido y acceso lineal\n";
    Tensor<int,2> t2(2,3);
    t2.fill(1);
    t2.reshape({3,2});
    int y = t2[5];
    int z = t2(2,1);
    assert(y == z);
    std::cout << "✓ y == t2(2,1): " << y << " == " << z << "\n\n";
}

void test_case_3() {
    std::cout << "Test Case 3: Reshape inválido\n";
    Tensor<int,3> t3(2,2,2);
    try {
        t3.reshape({2,4,1});
        std::cout << "✗ Exception not thrown\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        std::cout << "✓ std::invalid_argument thrown\n\n";
    }
}

void test_case_4() {
    std::cout << "Test Case 4: Suma y resta de tensores\n";
    Tensor<double,2> a(2,2), b(2,2);
    a.fill(0.0);
    a(0,1) = 5.5;
    b.fill(2.0);
    auto sum = a + b;
    auto diff = sum - b;

    assert(std::abs(sum(0,1) - 7.5) < 1e-9);
    assert(std::abs(diff(0,1) - 5.5) < 1e-9);
    std::cout << "✓ sum(0,1) == 7.5: " << sum(0,1) << "\n";
    std::cout << "✓ diff(0,1) == 5.5: " << diff(0,1) << "\n\n";
}

void test_case_5() {
    std::cout << "Test Case 5: Multiplicación escalar y tensores 3D\n";
    Tensor<float,1> v(3);
    v.fill(2.0f);
    auto scaled = v * 4.0f;

    Tensor<int,3> cube(2,2,2);
    cube.fill(1);
    auto cube2 = cube * cube;

    assert(std::abs(scaled(2) - 8.0f) < 1e-6);
    assert(cube2(1,1,1) == 1);
    std::cout << "✓ scaled(2) == 8.0f: " << scaled(2) << "\n";
    std::cout << "✓ cube2(1,1,1) == 1: " << cube2(1,1,1) << "\n\n";
}

void test_case_6() {
    std::cout << "Test Case 6: Broadcasting implícito\n";
    Tensor<int,2> m(2,1);
    m(0,0) = 3; m(1,0) = 4;
    Tensor<int,2> n(2,3);
    n.fill(5);
    auto p = m * n;

    assert(p(0,2) == 15);
    assert(p(1,1) == 20);
    std::cout << "✓ p(0,2) == 15: " << p(0,2) << "\n";
    std::cout << "✓ p(1,1) == 20: " << p(1,1) << "\n\n";
}

void test_case_7() {
    std::cout << "Test Case 7: Transpose 2D\n";
    Tensor<int,2> m2(2,3);
    m2.fill(0);
    m2(1,0) = 42;
    auto mt = m2.transpose_2d();

    auto expected_shape = std::array<size_t,2>{3,2};
    assert(mt.shape() == expected_shape);
    assert(mt(0,1) == m2(1,0));
    std::cout << "✓ mt.shape() == {3,2}\n";
    std::cout << "✓ mt(0,1) == m2(1,0): " << mt(0,1) << " == " << m2(1,0) << "\n\n";
}

int main() {
    std::cout << "=== TENSOR TESTS ===\n\n";

    test_case_1();
    test_case_2();
    test_case_3();
    test_case_4();
    test_case_5();
    test_case_6();
    test_case_7();

    std::cout << "All tensor tests passed! ✓\n";
    return 0;
}
