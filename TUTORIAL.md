# Tutorial Completo - Puzzle #71 CUDA Searcher

## O que esse programa faz?

Ele busca a chave privada do Bitcoin Puzzle #71 usando sua RTX 3050.
A chave está em algum lugar no range:
- Início: `0x400000000000000000`
- Fim:    `0x7fffffffffffffffff`
- Total:  ~1.18 × 10²¹ possibilidades (número gigante)

**Técnicas matemáticas usadas (o que nos dá ~4x de velocidade grátis):**
1. **Endomorphism SECP256K1**: cada ponto calculado gera um segundo ponto
   "reflexo" (beta·x, y) sem custo de multiplicação EC
2. **Negação de ponto**: inverte Y do ponto → endereço diferente, grátis
3. **Busca aleatória paralela**: cada thread da GPU cobre uma região diferente
   sem repetir o trabalho de outra thread

---

## PASSO 1 — Verificar pré-requisitos

Abra o WSL2 e rode esses comandos um por um:

```bash
# Verifica se o CUDA Toolkit está instalado
nvcc --version
```
Deve aparecer: `release 12.0` (ou similar)

```bash
# Verifica se a GPU está visível
nvidia-smi
```
Deve aparecer: `NVIDIA GeForce RTX 3050`

```bash
# Verifica se o make está instalado
make --version
```
Se não tiver, instale:
```bash
sudo apt update && sudo apt install -y build-essential
```

---

## PASSO 2 — Criar a pasta do projeto

```bash
mkdir -p ~/puzzle71
```
> Cria a pasta `puzzle71` dentro do seu diretório home.

```bash
cd ~/puzzle71
```
> Entra na pasta.

---

## PASSO 3 — Copiar os arquivos

Você precisa de 3 arquivos dentro de `~/puzzle71/`:
- `secp256k1.cuh`  → matemática da curva elíptica
- `kernel.cu`      → código que roda na GPU
- `main.cpp`       → código que roda na CPU (controla a GPU)
- `Makefile`       → instruções de compilação

Copie cada arquivo para a pasta. Se você recebeu os arquivos prontos,
só mova eles:
```bash
ls ~/puzzle71/
```
Deve listar os 4 arquivos acima.

---

## PASSO 4 — Compilar

```bash
cd ~/puzzle71
```

```bash
make
```

> O compilador `nvcc` vai juntar os arquivos e criar o executável `puzzle71`.
> Isso leva de 30 segundos a 2 minutos na primeira vez.

**Se der erro de arquitetura GPU**, edite o Makefile e troque `sm_86` por `sm_75`:
```bash
nano Makefile
```
Procure a linha `ARCH = -gencode arch=compute_86,code=sm_86`
e troque para:
`ARCH = -gencode arch=compute_75,code=sm_75`

Depois compile de novo:
```bash
make clean && make
```

---

## PASSO 5 — Executar

```bash
cd ~/puzzle71
```

```bash
./puzzle71
```

Você vai ver algo assim:
```
╔══════════════════════════════════════════════════════════╗
║         Puzzle #71 CUDA Searcher - EndoMorphism         ║
╚══════════════════════════════════════════════════════════╝

[GPU] NVIDIA GeForce RTX 3050 | SM: 8.6 | VRAM: 4096 MB | SMs: 20
[CONFIG] Blocos: 160 | Threads/bloco: 256 | Threads totais: 40960
[BUSCA] Iniciando... (Ctrl+C para pausar e salvar)
─────────────────────────────────────────────────────────────
[42s] Testadas: 3.210e+09 chaves | Velocidade: 76.43 Mkeys/s | Launches: 25
```

---

## Como pausar sem perder progresso

Pressione **Ctrl+C** a qualquer momento.

O programa vai:
1. Detectar o sinal
2. Salvar o checkpoint em `puzzle71.ckpt`
3. Encerrar com segurança

Para retomar de onde parou:
```bash
./puzzle71
```
Ele detecta o arquivo `puzzle71.ckpt` e continua automaticamente.

---

## Se encontrar a chave

O programa vai exibir na tela:
```
╔══════════════════════════════════════════════════════════╗
║          *** CHAVE PRIVADA ENCONTRADA! ***               ║
╠══════════════════════════════════════════════════════════╣
║  HEX: 000000000000000000...xxxxxxxxxxxxxxxx              ║
╚══════════════════════════════════════════════════════════╝
```

E salvar em `FOUND_KEY.txt`.

---

## Comandos de referência rápida

| O que fazer | Comando |
|---|---|
| Compilar | `make` |
| Executar | `./puzzle71` |
| Limpar compilação | `make clean` |
| Ver checkpoint | `cat puzzle71.ckpt \| xxd` |
| Apagar checkpoint | `rm puzzle71.ckpt` |
| Ver se GPU está ativa | `nvidia-smi` em outro terminal |
| Monitorar GPU em tempo real | `watch -n1 nvidia-smi` |

---

## Velocidade esperada na RTX 3050

| Métrica | Valor estimado |
|---|---|
| Multiplicações EC/s | ~20 Mkeys/s |
| Com endomorphism (×4) | ~80 Mkeys/s |
| Cobertura do range #71 em 1 dia | ~0.0006% |

> O puzzle #71 tem ~10²¹ chaves. Na velocidade de 80 Mkeys/s,
> cobrir o range inteiro levaria astronomicamente muito tempo.
> A busca é probabilística (aleatória): você pode encontrar na
> primeira hora ou nunca. É como uma loteria computacional.

---

## Resolução de problemas comuns

**Erro: `nvcc: command not found`**
```bash
export PATH=/usr/local/cuda/bin:$PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Erro: `no CUDA-capable device is detected`**
- No WSL2, confirme que está usando Windows 11 ou Windows 10 21H2+
- Instale o driver NVIDIA for WSL2 (não o driver Linux padrão)
- Verifique: `ls /dev/dxg` (deve existir)

**Erro: `sm_86` não suportado**
- Edite o Makefile e use `sm_75`

**O programa compila mas trava na GPU**
- Reduza `num_blocks` no `main.cpp`: troque `* 8` por `* 4`
- Recompile com `make clean && make`

---

## Estrutura dos arquivos

```
puzzle71/
├── secp256k1.cuh   ← Matemática: campo finito, ponto EC, endomorphism
├── kernel.cu       ← Kernel GPU: busca, hash160, 4 candidatos por key
├── main.cpp        ← CPU: lança kernel, checkpoint, display
├── Makefile        ← Como compilar
├── puzzle71.ckpt   ← Criado ao pausar (checkpoint)
└── FOUND_KEY.txt   ← Criado SE encontrar a chave
```
