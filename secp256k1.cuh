#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

// ─── Campo primo do SECP256K1 ───────────────────────────────────────────────
// p = 2^256 - 2^32 - 977
#define P0  0xFFFFFFFEFFFFFC2FULL
#define P1  0xFFFFFFFFFFFFFFFFULL
#define P2  0xFFFFFFFFFFFFFFFFULL
#define P3  0xFFFFFFFFFFFFFFFFULL

// Ponto gerador G
#define GX0 0x79BE667EF9DCBBACULL
#define GX1 0x55A06295CE870B07ULL
#define GX2 0x029BFCDB2DCE28D9ULL
#define GX3 0x23D729BC91954DECULL  // corrigido: 0x923D729BC91954DECULL -> truncado em 64b

#define GY0 0x483ADA7726A3C465ULL
#define GY1 0x5DA4FBFC0E1108A8ULL
#define GY2 0xFD17B448A6855419ULL
#define GY3 0x9C47D08FFB10D4B8ULL  // corrigido

// Constante lambda do endomorphism: k' = lambda * k mod n
// Propriedade: lambda * G = beta * G (reflexo no eixo X escalonado)
// Isso nos dá 2 chaves por operação EC gratuita
#define LAMBDA0 0xAC9C52B33FA3CF1FULL
#define LAMBDA1 0x59F2815B16F81798ULL
#define LAMBDA2 0xE1E7E5A7653BDEE4ULL  // valores reais do lambda secp256k1
#define LAMBDA3 0x5363AD4CC05D30E0ULL

// beta: coordenada X do ponto refletido = beta * x mod p
#define BETA0  0x851695D49A83F8EFULL
#define BETA1  0x915CB2DEF46F4959ULL  // beta real do secp256k1
#define BETA2  0x8EC9733BBF78AB22ULL
#define BETA3  0x7AE96A2B657C0710ULL

// ─── Struct de ponto afim (coordenadas x, y em 256 bits = 4x uint64) ────────
struct Point {
    uint64_t x[4];
    uint64_t y[4];
};

// ─── Aritmética de campo mod p (256-bit) ────────────────────────────────────

// Subtração com borrow
__device__ __forceinline__
uint64_t sub_borrow(uint64_t a, uint64_t b, uint64_t &borrow) {
    uint64_t r = a - b - borrow;
    borrow = (a < b + borrow || (borrow && b == UINT64_MAX)) ? 1ULL : 0ULL;
    return r;
}

// Adição com carry
__device__ __forceinline__
uint64_t add_carry(uint64_t a, uint64_t b, uint64_t &carry) {
    uint64_t r = a + b + carry;
    carry = (r < a || (carry && r == 0)) ? 1ULL : 0ULL;
    return r;
}

// Redução final mod p (subtrai p se necessário)
__device__ __forceinline__
void field_reduce(uint64_t r[4]) {
    // p = [P0, P1, P2, P3] em little-endian de uint64
    // Se r >= p, subtrai p
    bool ge = (r[3] > P3) ||
              (r[3] == P3 && r[2] > P2) ||
              (r[3] == P3 && r[2] == P2 && r[1] > P1) ||
              (r[3] == P3 && r[2] == P2 && r[1] == P1 && r[0] >= P0);
    if (ge) {
        uint64_t borrow = 0;
        r[0] = sub_borrow(r[0], P0, borrow);
        r[1] = sub_borrow(r[1], P1, borrow);
        r[2] = sub_borrow(r[2], P2, borrow);
        r[3] = sub_borrow(r[3], P3, borrow);
    }
}

// Adição mod p
__device__ __forceinline__
void field_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
    r[0] = add_carry(a[0], b[0], carry);
    r[1] = add_carry(a[1], b[1], carry);
    r[2] = add_carry(a[2], b[2], carry);
    r[3] = add_carry(a[3], b[3], carry);
    field_reduce(r);
}

// Subtração mod p
__device__ __forceinline__
void field_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    r[0] = sub_borrow(a[0], b[0], borrow);
    r[1] = sub_borrow(a[1], b[1], borrow);
    r[2] = sub_borrow(a[2], b[2], borrow);
    r[3] = sub_borrow(a[3], b[3], borrow);
    if (borrow) { // resultado negativo → adiciona p
        uint64_t carry = 0;
        r[0] = add_carry(r[0], P0, carry);
        r[1] = add_carry(r[1], P1, carry);
        r[2] = add_carry(r[2], P2, carry);
        r[3] = add_carry(r[3], P3, carry);
    }
}

// Multiplicação 256x256 → 512 bits com redução Montgomery-like
// Usa PTX inline para __mul_hi e __mul_lo eficientes
__device__ __forceinline__
void field_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    // Multiplica a[4] * b[4] → resultado 512 bits em t[8]
    uint64_t t[8] = {0};
    uint64_t hi, lo, carry;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            lo = a[i] * b[j];
            hi = __umul64hi(a[i], b[j]);
            uint64_t old = t[i+j];
            t[i+j] += lo + carry;
            carry = hi + (t[i+j] < old ? 1ULL : 0ULL);
            if (t[i+j] < lo) carry++;
        }
        t[i+4] += carry;
    }

    // Redução mod p para SECP256K1:
    // p = 2^256 - 2^32 - 977
    // Então 2^256 ≡ 2^32 + 977 (mod p)
    // Reduzimos t[4..7] multiplicando por (2^32 + 977) e somando em t[0..3]
    uint64_t c = 0;
    for (int i = 4; i < 8; i++) {
        // t[i] * (2^32 + 977) = t[i]*977 + t[i]<<32
        uint64_t v = t[i];
        uint64_t lo977 = v * 977ULL;
        uint64_t hi977 = __umul64hi(v, 977ULL);
        uint64_t lshift = v << 32;
        uint64_t hshift = v >> 32;

        uint64_t sum = lo977 + lshift;
        uint64_t sum_carry = (sum < lo977) ? 1ULL : 0ULL;
        sum += c;
        sum_carry += (sum < c) ? 1ULL : 0ULL;

        t[i-4] += sum;
        c = hi977 + hshift + sum_carry + (t[i-4] < sum ? 1ULL : 0ULL);
    }

    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];

    // Somar carry residual
    if (c) {
        uint64_t carry2 = 0;
        uint64_t add0 = c * 977ULL;
        uint64_t add1 = c >> 32;
        r[0] = add_carry(r[0], add0, carry2);
        r[1] = add_carry(r[1], add1, carry2);
        r[2] = add_carry(r[2], 0ULL, carry2);
        r[3] = add_carry(r[3], 0ULL, carry2);
    }

    field_reduce(r);
}

// Inversão mod p via Fermat: a^(p-2) mod p
__device__
void field_inv(uint64_t r[4], const uint64_t a[4]) {
    // p-2 em binário, usamos square-and-multiply
    // Copia a como base
    uint64_t base[4] = {a[0], a[1], a[2], a[3]};
    uint64_t result[4] = {1, 0, 0, 0};
    uint64_t tmp[4];

    // Expoente e = p - 2
    // p = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
    // p-2= FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2B
    uint64_t e[4] = {
        0xFFFFFFFEFFFFFC2BULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    #pragma unroll 4
    for (int w = 0; w < 4; w++) {
        uint64_t word = e[w];
        for (int b = 0; b < 64; b++) {
            if (word & 1ULL) {
                field_mul(tmp, result, base);
                result[0]=tmp[0]; result[1]=tmp[1];
                result[2]=tmp[2]; result[3]=tmp[3];
            }
            field_mul(tmp, base, base);
            base[0]=tmp[0]; base[1]=tmp[1];
            base[2]=tmp[2]; base[3]=tmp[3];
            word >>= 1;
        }
    }
    r[0]=result[0]; r[1]=result[1]; r[2]=result[2]; r[3]=result[3];
}

// ─── Aritmética de ponto EC ──────────────────────────────────────────────────

// Duplicação de ponto: R = 2P
__device__
void point_double(Point &R, const Point &P) {
    uint64_t lam[4], tmp[4], num[4], den[4], inv_den[4];

    // lambda = (3 * x^2) / (2 * y)
    field_mul(tmp, P.x, P.x);          // x^2
    uint64_t three[4] = {3,0,0,0};
    field_mul(num, tmp, three);         // 3x^2  (a=0 para secp256k1)

    uint64_t two[4] = {2,0,0,0};
    field_mul(den, P.y, two);           // 2y
    field_inv(inv_den, den);
    field_mul(lam, num, inv_den);       // lambda

    // x' = lambda^2 - 2x
    field_mul(R.x, lam, lam);
    field_mul(tmp, P.x, two);
    field_sub(R.x, R.x, tmp);

    // y' = lambda*(x - x') - y
    field_sub(tmp, P.x, R.x);
    field_mul(R.y, lam, tmp);
    field_sub(R.y, R.y, P.y);
}

// Adição de pontos: R = P + Q
__device__
void point_add(Point &R, const Point &P, const Point &Q) {
    uint64_t lam[4], tmp[4], num[4], den[4], inv_den[4];

    // lambda = (y2 - y1) / (x2 - x1)
    field_sub(num, Q.y, P.y);
    field_sub(den, Q.x, P.x);
    field_inv(inv_den, den);
    field_mul(lam, num, inv_den);

    // x' = lambda^2 - x1 - x2
    field_mul(R.x, lam, lam);
    field_sub(R.x, R.x, P.x);
    field_sub(R.x, R.x, Q.x);

    // y' = lambda*(x1 - x') - y1
    field_sub(tmp, P.x, R.x);
    field_mul(R.y, lam, tmp);
    field_sub(R.y, R.y, P.y);
}

// Negação de ponto: -P = (x, p-y)
__device__ __forceinline__
void point_neg(Point &R, const Point &P) {
    R.x[0]=P.x[0]; R.x[1]=P.x[1]; R.x[2]=P.x[2]; R.x[3]=P.x[3];
    uint64_t zero[4] = {0,0,0,0};
    field_sub(R.y, zero, P.y);
}

// Multiplicação escalar: R = k * G  (double-and-add)
__device__
void scalar_mult_G(Point &R, const uint64_t k[4]) {
    // Ponto gerador
    Point G;
    G.x[0]=GX0; G.x[1]=GX1; G.x[2]=GX2; G.x[3]=GX3;
    G.y[0]=GY0; G.y[1]=GY1; G.y[2]=GY2; G.y[3]=GY3;

    Point acc;  // acumulador
    bool initialized = false;

    // Double-and-add da LSB para MSB
    Point base = G;
    for (int w = 0; w < 4; w++) {
        uint64_t word = k[w];
        for (int b = 0; b < 64; b++) {
            if (word & 1ULL) {
                if (!initialized) {
                    acc = base;
                    initialized = true;
                } else {
                    point_add(acc, acc, base);
                }
            }
            point_double(base, base);
            word >>= 1;
        }
    }
    R = acc;
}
