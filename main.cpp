// puzzle71 - Main CPU
// Controla o kernel CUDA, checkpoint/resume, display de progresso

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <signal.h>

// ─── Declaração do kernel (definido em kernel.cu) ───────────────────────────
__global__ void search_kernel(
    const uint64_t range_start_lo,
    const uint64_t range_start_hi,
    const uint64_t range_size_lo,
    const uint64_t range_size_hi,
    const uint8_t  target_hash[20],
    uint64_t *found_key,
    uint32_t *found_flag,
    uint64_t *total_checked,
    const uint64_t seed,
    const uint64_t keys_per_thread
);

// ─── Alvo: hash160 de 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU ──────────────────
// Calculado offline: RIPEMD160(SHA256(pubkey))
// Endereço: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
static const uint8_t TARGET_HASH160[20] = {
    0xf6, 0xf5, 0x43, 0x1d, 0x25, 0xbb, 0xf7, 0xb1,
    0x2e, 0x8a, 0xdd, 0x9a, 0xf5, 0xe3, 0x47, 0x5c,
    0x44, 0xa0, 0xa5, 0xb8
};

// ─── Range do puzzle #71 ────────────────────────────────────────────────────
// 0x400000000000000000 : 0x7fffffffffffffffff
// = 2^70 : 2^71 - 1
static const uint64_t RANGE_START_LO = 0x0000000000000000ULL;
static const uint64_t RANGE_START_HI = 0x0000000000000040ULL; // bit 70 = 0x40 no uint64 alto
static const uint64_t RANGE_SIZE_LO  = 0xFFFFFFFFFFFFFFFFULL;
static const uint64_t RANGE_SIZE_HI  = 0x000000000000003FULL;

// ─── Checkpoint ─────────────────────────────────────────────────────────────
#define CHECKPOINT_FILE "puzzle71.ckpt"

typedef struct {
    uint64_t total_checked;
    uint64_t last_seed;
    double   elapsed_seconds;
    char     magic[8]; // "P71CKPT\0"
} Checkpoint;

void save_checkpoint(uint64_t checked, uint64_t seed, double elapsed) {
    FILE *f = fopen(CHECKPOINT_FILE, "wb");
    if (!f) { fprintf(stderr, "[AVISO] Nao foi possivel salvar checkpoint\n"); return; }
    Checkpoint ck;
    ck.total_checked = checked;
    ck.last_seed = seed;
    ck.elapsed_seconds = elapsed;
    memcpy(ck.magic, "P71CKPT", 8);
    fwrite(&ck, sizeof(ck), 1, f);
    fclose(f);
    printf("\n[CHECKPOINT] Salvo: %.2e chaves testadas | Semente: %llu\n",
           (double)checked, (unsigned long long)seed);
}

int load_checkpoint(uint64_t *checked, uint64_t *seed, double *elapsed) {
    FILE *f = fopen(CHECKPOINT_FILE, "rb");
    if (!f) return 0;
    Checkpoint ck;
    fread(&ck, sizeof(ck), 1, f);
    fclose(f);
    if (memcmp(ck.magic, "P71CKPT", 7) != 0) return 0;
    *checked = ck.total_checked;
    *seed    = ck.last_seed;
    *elapsed = ck.elapsed_seconds;
    return 1;
}

// ─── Sinal para Ctrl+C → salva checkpoint antes de sair ────────────────────
static volatile int g_stop = 0;
static uint64_t g_checked  = 0;
static uint64_t g_seed     = 0;
static double   g_elapsed  = 0.0;

void handle_sigint(int sig) {
    (void)sig;
    g_stop = 1;
    printf("\n[INFO] Ctrl+C detectado. Salvando checkpoint...\n");
    save_checkpoint(g_checked, g_seed, g_elapsed);
    exit(0);
}

// ─── Converte chave uint64[4] → hex string ──────────────────────────────────
void key_to_hex(const uint64_t k[4], char *out) {
    // k[3] é o limb mais significativo
    sprintf(out, "%016llx%016llx%016llx%016llx",
            (unsigned long long)k[3],
            (unsigned long long)k[2],
            (unsigned long long)k[1],
            (unsigned long long)k[0]);
}

// ─── Salva resultado encontrado ──────────────────────────────────────────────
void save_found(const uint64_t k[4]) {
    char hex[70] = {0};
    key_to_hex(k, hex);
    printf("\n\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║          *** CHAVE PRIVADA ENCONTRADA! ***               ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  HEX: %s  ║\n", hex);
    printf("╚══════════════════════════════════════════════════════════╝\n");

    FILE *f = fopen("FOUND_KEY.txt", "w");
    if (f) {
        fprintf(f, "Bitcoin Puzzle #71 - Chave Encontrada!\n");
        fprintf(f, "Chave Privada (HEX): %s\n", hex);
        fprintf(f, "Endereco Alvo: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU\n");
        fclose(f);
        printf("[INFO] Salvo em FOUND_KEY.txt\n");
    }
}

// ─── main ────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║         Puzzle #71 CUDA Searcher - EndoMorphism         ║\n");
    printf("║   Tecnica: Endomorphism + Negacao + Busca Aleatoria     ║\n");
    printf("║   Alvo: 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ── Verifica GPU ─────────────────────────────────────────────────────────
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "[ERRO] Nenhuma GPU CUDA encontrada!\n");
        fprintf(stderr, "Verifique: nvidia-smi e nvcc --version\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("[GPU] %s | SM: %d.%d | VRAM: %zu MB | SMs: %d\n",
           prop.name,
           prop.major, prop.minor,
           prop.totalGlobalMem / (1024*1024),
           prop.multiProcessorCount);

    // ── Configuração de threads ───────────────────────────────────────────────
    // RTX 3050: 20 SMs × 128 = 2560 CUDA cores
    // Usamos 256 threads por bloco, 128 blocos → 32768 threads simultâneas
    int threads_per_block = 256;
    int num_blocks        = prop.multiProcessorCount * 8; // 8 blocos por SM
    uint64_t keys_per_thread = 1000; // cada thread testa 1000 chaves por lançamento

    printf("[CONFIG] Blocos: %d | Threads/bloco: %d | Threads totais: %d\n",
           num_blocks, threads_per_block, num_blocks * threads_per_block);
    printf("[CONFIG] Chaves por thread por lancamento: %llu\n",
           (unsigned long long)keys_per_thread);
    printf("[SPEED]  ~%llu chaves/lancamento (x4 com endomorphism)\n\n",
           (unsigned long long)num_blocks * threads_per_block * keys_per_thread * 4);

    // ── Aloca memória GPU ─────────────────────────────────────────────────────
    uint8_t  *d_target;
    uint64_t *d_found_key;
    uint32_t *d_found_flag;
    uint64_t *d_total_checked;

    cudaMalloc(&d_target,        20 * sizeof(uint8_t));
    cudaMalloc(&d_found_key,     4  * sizeof(uint64_t));
    cudaMalloc(&d_found_flag,    1  * sizeof(uint32_t));
    cudaMalloc(&d_total_checked, 1  * sizeof(uint64_t));

    cudaMemcpy(d_target, TARGET_HASH160, 20, cudaMemcpyHostToDevice);

    uint32_t zero32 = 0;
    uint64_t zero64 = 0;
    cudaMemcpy(d_found_flag,    &zero32, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_checked, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    uint64_t found_key_init[4] = {0,0,0,0};
    cudaMemcpy(d_found_key, found_key_init, 4*sizeof(uint64_t), cudaMemcpyHostToDevice);

    // ── Checkpoint/Resume ─────────────────────────────────────────────────────
    uint64_t total_checked = 0;
    double   elapsed_acc   = 0.0;
    uint64_t current_seed  = (uint64_t)time(NULL);

    if (load_checkpoint(&total_checked, &current_seed, &elapsed_acc)) {
        printf("[RESUME] Checkpoint encontrado!\n");
        printf("[RESUME] Chaves ja testadas: %.2e\n", (double)total_checked);
        printf("[RESUME] Retomando da semente: %llu\n\n", (unsigned long long)current_seed);
        current_seed++; // avança semente para não repetir exatamente
    } else {
        printf("[INICIO] Nenhum checkpoint. Iniciando busca nova.\n\n");
    }

    // ── Sinal Ctrl+C ──────────────────────────────────────────────────────────
    signal(SIGINT, handle_sigint);

    // ── Loop principal ────────────────────────────────────────────────────────
    printf("[BUSCA] Iniciando... (Ctrl+C para pausar e salvar)\n");
    printf("─────────────────────────────────────────────────────────────\n");

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    uint64_t launch_count = 0;
    uint64_t checkpoint_interval = 50; // salva checkpoint a cada 50 lançamentos

    while (!g_stop) {
        // Lança kernel
        search_kernel<<<num_blocks, threads_per_block>>>(
            RANGE_START_LO,
            RANGE_START_HI,
            RANGE_SIZE_LO,
            RANGE_SIZE_HI,
            d_target,
            d_found_key,
            d_found_flag,
            d_total_checked,
            current_seed,
            keys_per_thread
        );

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ERRO CUDA] %s\n", cudaGetErrorString(err));
            break;
        }

        // Verifica se encontrou
        uint32_t found_flag = 0;
        cudaMemcpy(&found_flag, d_found_flag, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (found_flag) {
            uint64_t found_key[4];
            cudaMemcpy(found_key, d_found_key, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
            save_found(found_key);
            save_checkpoint(total_checked, current_seed, g_elapsed);
            break;
        }

        // Atualiza contador
        uint64_t gpu_checked = 0;
        cudaMemcpy(&gpu_checked, d_total_checked, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        uint64_t this_batch = (uint64_t)num_blocks * threads_per_block * keys_per_thread * 4;
        total_checked += this_batch;
        g_checked = total_checked;

        // Tempo e velocidade
        clock_gettime(CLOCK_MONOTONIC, &t_now);
        double dt = (t_now.tv_sec - t_start.tv_sec) +
                    (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
        double total_time = elapsed_acc + dt;
        g_elapsed = total_time;
        double speed = (double)total_checked / total_time;

        // Display
        launch_count++;
        if (launch_count % 5 == 0) { // atualiza a cada 5 lançamentos
            printf("\r[%.0fs] Testadas: %.3e chaves | Velocidade: %.2f Mkeys/s | Launches: %llu   ",
                   total_time,
                   (double)total_checked,
                   speed / 1e6,
                   (unsigned long long)launch_count);
            fflush(stdout);
        }

        // Checkpoint periódico
        if (launch_count % checkpoint_interval == 0) {
            g_seed = current_seed;
            save_checkpoint(total_checked, current_seed, total_time);
        }

        // Avança semente
        current_seed += (uint64_t)num_blocks * threads_per_block + 1;
        g_seed = current_seed;

        // Reseta contador GPU
        cudaMemcpy(d_total_checked, &zero64, sizeof(uint64_t), cudaMemcpyHostToDevice);
    }

    // ── Limpeza ───────────────────────────────────────────────────────────────
    cudaFree(d_target);
    cudaFree(d_found_key);
    cudaFree(d_found_flag);
    cudaFree(d_total_checked);

    printf("\n[FIM] Encerrando.\n");
    return 0;
}
