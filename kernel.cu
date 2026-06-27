// puzzle71 - Buscador CUDA para Bitcoin Puzzle #71
// Técnicas: endomorphism secp256k1 + negação de ponto + busca aleatória paralela
// Alvo: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
// Range: 0x400000000000000000 : 0x7fffffffffffffffff

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "secp256k1.cuh"

// ─── Hash SHA-256 (implementação compacta para GPU) ─────────────────────────
__constant__ uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
    0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
    0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
    0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
    0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
    0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(e,f,g)   (((e)&(f))^(~(e)&(g)))
#define MAJ(a,b,c)  (((a)&(b))^((a)&(c))^((b)&(c)))
#define EP0(a)      (ROTR32(a,2)^ROTR32(a,13)^ROTR32(a,22))
#define EP1(e)      (ROTR32(e,6)^ROTR32(e,11)^ROTR32(e,25))
#define SIG0(x)     (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define SIG1(x)     (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

__device__
void sha256_block(uint32_t state[8], const uint32_t data[16]) {
    uint32_t w[64], a,b,c,d,e,f,g,h,t1,t2;
    for(int i=0;i<16;i++) w[i]=data[i];
    for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    a=state[0];b=state[1];c=state[2];d=state[3];
    e=state[4];f=state[5];g=state[6];h=state[7];
    for(int i=0;i<64;i++){
        t1=h+EP1(e)+CH(e,f,g)+K256[i]+w[i];
        t2=EP0(a)+MAJ(a,b,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

// SHA256 de 33 bytes (chave pública comprimida)
__device__
void sha256_33(const uint8_t *in, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };
    uint32_t block[16] = {0};
    // Copia 33 bytes (big-endian nos uint32)
    uint8_t buf[64] = {0};
    for(int i=0;i<33;i++) buf[i]=in[i];
    buf[33]=0x80; // padding
    // comprimento em bits = 33*8 = 264 → big-endian nos últimos 8 bytes
    buf[62]=0x01; buf[63]=0x08;
    for(int i=0;i<16;i++)
        block[i] = ((uint32_t)buf[i*4]<<24)|((uint32_t)buf[i*4+1]<<16)|
                   ((uint32_t)buf[i*4+2]<<8)|(uint32_t)buf[i*4+3];
    sha256_block(state, block);
    for(int i=0;i<8;i++){
        out[i*4]  =(state[i]>>24)&0xFF;
        out[i*4+1]=(state[i]>>16)&0xFF;
        out[i*4+2]=(state[i]>>8)&0xFF;
        out[i*4+3]=(state[i])&0xFF;
    }
}

// ─── RIPEMD-160 ─────────────────────────────────────────────────────────────
#define ROTL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
#define F1(x,y,z) ((x)^(y)^(z))
#define F2(x,y,z) (((x)&(y))|(~(x)&(z)))
#define F3(x,y,z) (((x)|(~(y)))^(z))
#define F4(x,y,z) (((x)&(z))|((y)&(~(z))))
#define F5(x,y,z) ((x)^((y)|(~(z))))

__constant__ uint32_t RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
__constant__ uint32_t RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
__constant__ uint32_t SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
__constant__ uint32_t SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};
__constant__ uint32_t KL[5] = {0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
__constant__ uint32_t KR[5] = {0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000};

__device__
void ripemd160_32(const uint8_t *in, uint8_t out[20]) {
    uint32_t h0=0x67452301,h1=0xEFCDAB89,h2=0x98BADCFE,h3=0x10325476,h4=0xC3D2E1F0;
    uint32_t x[16]={0};
    uint8_t buf[64]={0};
    for(int i=0;i<32;i++) buf[i]=in[i];
    buf[32]=0x80;
    uint32_t bitlen=256;
    buf[56]=(bitlen)&0xFF; buf[57]=0; buf[58]=0; buf[59]=0;
    for(int i=0;i<16;i++)
        x[i]=((uint32_t)buf[i*4+3]<<24)|((uint32_t)buf[i*4+2]<<16)|
             ((uint32_t)buf[i*4+1]<<8)|(uint32_t)buf[i*4];
    uint32_t al=h0,bl=h1,cl=h2,dl=h3,el=h4;
    uint32_t ar=h0,br=h1,cr=h2,dr=h3,er=h4;
    uint32_t t,fl,fr;
    for(int i=0;i<80;i++){
        int r=i/16;
        if(r==0){fl=F1(bl,cl,dl); fr=F5(br,cr,dr);}
        else if(r==1){fl=F2(bl,cl,dl); fr=F4(br,cr,dr);}
        else if(r==2){fl=F3(bl,cl,dl); fr=F3(br,cr,dr);}
        else if(r==3){fl=F4(bl,cl,dl); fr=F2(br,cr,dr);}
        else{fl=F5(bl,cl,dl); fr=F1(br,cr,dr);}
        t=ROTL32(al+fl+x[RL[i]]+KL[r],SL[i])+el;
        al=el;el=dl;dl=ROTL32(cl,10);cl=bl;bl=t;
        t=ROTL32(ar+fr+x[RR[i]]+KR[r],SR[i])+er;
        ar=er;er=dr;dr=ROTL32(cr,10);cr=br;br=t;
    }
    t=h1+cl+dr; h1=h2+dl+er; h2=h3+el+ar;
    h3=h4+al+br; h4=h0+bl+cr; h0=t;
    // saída little-endian
    uint32_t hh[5]={h0,h1,h2,h3,h4};
    for(int i=0;i<5;i++){
        out[i*4]  = hh[i]&0xFF;
        out[i*4+1]=(hh[i]>>8)&0xFF;
        out[i*4+2]=(hh[i]>>16)&0xFF;
        out[i*4+3]=(hh[i]>>24)&0xFF;
    }
}

// ─── Ponto → Hash160 (RIPEMD160(SHA256(pubkey_comprimida))) ────────────────
__device__
void point_to_hash160(const Point &P, uint8_t hash[20]) {
    // Monta chave pública comprimida: 0x02 ou 0x03 + x (32 bytes big-endian)
    uint8_t pub[33];
    pub[0] = (P.y[0] & 1) ? 0x03 : 0x02;  // paridade do y
    // x em big-endian (P.x está em limbs little-endian de 64 bits)
    for(int i=0;i<8;i++){
        uint64_t limb = P.x[3-i/2];
        int shift = (1-(i%2))*32;
        uint32_t part = (limb >> shift) & 0xFFFFFFFF;
        pub[1+i*4]   = (part>>24)&0xFF;
        pub[1+i*4+1] = (part>>16)&0xFF;
        pub[1+i*4+2] = (part>>8)&0xFF;
        pub[1+i*4+3] = part&0xFF;
    }
    uint8_t sha[32];
    sha256_33(pub, sha);
    ripemd160_32(sha, hash);
}

// ─── Gerador congruencial para chaves aleatórias por thread ─────────────────
__device__ __forceinline__
void next_key(uint64_t k[4], uint64_t &state) {
    // LCG simples com período longo
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    k[0] ^= state;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    k[1] ^= state >> 3;
}

// ─── Kernel principal ────────────────────────────────────────────────────────
// Cada thread busca em uma sub-região aleatória do range [START, END]
// Por lote: computa k*G, (-k)*G, lambda*k*G, lambda*(-k)*G → 4 candidatos
// Isso usa o endomorphism: se P=k*G, então lambda*k*G = (beta*x, y)
// (reflexo no espaço de chaves, ponto diferente mas derivado gratuitamente)

__global__
void search_kernel(
    const uint64_t range_start_lo,  // 64 bits baixos do início do range
    const uint64_t range_start_hi,  // 64 bits altos do início do range
    const uint64_t range_size_lo,
    const uint64_t range_size_hi,
    const uint8_t  target_hash[20], // hash160 alvo
    uint64_t *found_key,            // saída: chave encontrada (4x uint64)
    uint32_t *found_flag,           // 1 se encontrou
    uint64_t *total_checked,        // contador de chaves testadas
    const uint64_t seed,            // semente aleatória por lançamento
    const uint64_t keys_per_thread  // quantas chaves cada thread testa
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Semente única por thread
    uint64_t rng_state = seed ^ ((uint64_t)tid * 0xDEADBEEFCAFEBABEULL);

    // Chave inicial: pega posição aleatória dentro do range
    // Range = [0x400000000000000000 : 0x7fffffffffffffffff]
    // = 0x400000000000000000 + rand() % 0x400000000000000000
    // Simplificado: usamos o tid + rng para distribuir uniformemente
    uint64_t k[4] = {0,0,0,0};
    // Bit 70 sempre setado (range começa em 2^70 = 0x400...0)
    // Range tem 70 bits de liberdade → bits 0-69 variam
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    k[0] = rng_state;
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    k[1] = rng_state & 0x3FULL; // apenas 6 bits (total 70 bits aleatórios)
    k[1] |= 0x40ULL;            // força bit 70 → início do range
    k[2] = 0; k[3] = 0;

    uint64_t checked = 0;

    for(uint64_t iter = 0; iter < keys_per_thread; iter++) {
        if (*found_flag) return; // outra thread já achou

        // ── Compute k*G ──────────────────────────────────────────────────
        Point P;
        scalar_mult_G(P, k);

        uint8_t h[20];

        // Candidato 1: k
        point_to_hash160(P, h);
        checked++;
        bool match = true;
        for(int i=0;i<20 && match;i++) match = (h[i]==target_hash[i]);
        if(match && atomicExch(found_flag,1)==0){
            found_key[0]=k[0]; found_key[1]=k[1];
            found_key[2]=k[2]; found_key[3]=k[3];
            return;
        }

        // Candidato 2: -k (negação de ponto → mesma chave pública X, Y invertido)
        // Endereço diferente pois y muda paridade
        Point Pneg;
        point_neg(Pneg, P);
        point_to_hash160(Pneg, h);
        checked++;
        match = true;
        for(int i=0;i<20 && match;i++) match = (h[i]==target_hash[i]);
        if(match && atomicExch(found_flag,1)==0){
            // chave negada = n - k (não implementamos aqui, sinalizamos)
            found_key[0]=k[0]; found_key[1]=k[1];
            found_key[2]=k[2]; found_key[3]=k[3];
            found_key[3] |= 0x8000000000000000ULL; // flag: é a negada
            return;
        }

        // Candidato 3: endomorphism - aplica beta em x → ponto diferente
        // beta * x mod p dá o X de lambda*k*G, grátis sem multiplicação EC
        Point Pbeta;
        uint64_t beta[4] = {BETA0, BETA1, BETA2, BETA3};
        field_mul(Pbeta.x, P.x, beta);
        Pbeta.y[0]=P.y[0]; Pbeta.y[1]=P.y[1];
        Pbeta.y[2]=P.y[2]; Pbeta.y[3]=P.y[3];
        point_to_hash160(Pbeta, h);
        checked++;
        match = true;
        for(int i=0;i<20 && match;i++) match = (h[i]==target_hash[i]);
        if(match && atomicExch(found_flag,1)==0){
            found_key[0]=k[0]; found_key[1]=k[1];
            found_key[2]=k[2]; found_key[3]=k[3];
            found_key[3] |= 0x4000000000000000ULL; // flag: é o beta
            return;
        }

        // Candidato 4: beta negado
        point_neg(Pbeta, Pbeta);
        point_to_hash160(Pbeta, h);
        checked++;
        match = true;
        for(int i=0;i<20 && match;i++) match = (h[i]==target_hash[i]);
        if(match && atomicExch(found_flag,1)==0){
            found_key[0]=k[0]; found_key[1]=k[1];
            found_key[2]=k[2]; found_key[3]=k[3];
            found_key[3] |= 0xC000000000000000ULL; // flag: beta negada
            return;
        }

        // Próxima chave: avança aleatoriamente para não repetir
        next_key(k, rng_state);
        // Garante que k ainda está no range [2^70, 2^71-1]
        k[2] = 0; k[3] = 0;
        k[1] = (k[1] & 0x3FULL) | 0x40ULL; // mantém bits 64-70
    }

    atomicAdd((unsigned long long*)total_checked, (unsigned long long)checked);
}
