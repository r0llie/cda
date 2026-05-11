#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct Result {
    uint32_t found;
    uint32_t _pad;
    uint64_t nonce;
    uint32_t hash[8];
};

struct Config {
    uint8_t challenge[32]{};
    uint32_t difficulty[8]{};
    std::vector<int> gpus;
    uint32_t blocks = 131072;
    uint32_t threads = 256;
    uint32_t iters = 128;
    uint64_t base_nonce = 0;
    int seconds = 60;
};

#define CUDA_CHECK(x)                                                                            \
    do {                                                                                         \
        cudaError_t err__ = (x);                                                                 \
        if (err__ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

__device__ __forceinline__ uint32_t bswap32d(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

struct U64x {
    uint32_t lo;
    uint32_t hi;
};

__device__ __constant__ U64x KECCAK_RC[24] = {
    {0x00000001u, 0x00000000u}, {0x00008082u, 0x00000000u}, {0x0000808au, 0x80000000u},
    {0x80008000u, 0x80000000u}, {0x0000808bu, 0x00000000u}, {0x80000001u, 0x00000000u},
    {0x80008081u, 0x80000000u}, {0x00008009u, 0x80000000u}, {0x0000008au, 0x00000000u},
    {0x00000088u, 0x00000000u}, {0x80008009u, 0x00000000u}, {0x8000000au, 0x00000000u},
    {0x8000808bu, 0x00000000u}, {0x0000008bu, 0x80000000u}, {0x00008089u, 0x80000000u},
    {0x00008003u, 0x80000000u}, {0x00008002u, 0x80000000u}, {0x00000080u, 0x80000000u},
    {0x0000800au, 0x00000000u}, {0x8000000au, 0x80000000u}, {0x80008081u, 0x80000000u},
    {0x00008080u, 0x80000000u}, {0x80000001u, 0x00000000u}, {0x80008008u, 0x80000000u},
};

__device__ __forceinline__ U64x make_u64(uint32_t lo, uint32_t hi) {
    U64x v{lo, hi};
    return v;
}

__device__ __forceinline__ U64x xor64(U64x a, U64x b) {
    return make_u64(a.lo ^ b.lo, a.hi ^ b.hi);
}

__device__ __forceinline__ U64x andnot64(U64x a, U64x b) {
    return make_u64((~a.lo) & b.lo, (~a.hi) & b.hi);
}

__device__ __forceinline__ U64x rotl64(U64x v, uint32_t n) {
    n &= 63u;
    if (n == 0u) return v;
    if (n == 32u) return make_u64(v.hi, v.lo);
    if (n < 32u) {
        uint32_t m = 32u - n;
        return make_u64((v.lo << n) | (v.hi >> m), (v.hi << n) | (v.lo >> m));
    }
    uint32_t s = n - 32u;
    uint32_t m = 32u - s;
    return make_u64((v.hi << s) | (v.lo >> m), (v.lo << s) | (v.hi >> m));
}

__device__ __forceinline__ uint32_t load_le32(const uint8_t* p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ void keccak_f1600(U64x s[25]) {
    for (int r = 0; r < 24; r++) {
        U64x C0 = xor64(xor64(xor64(xor64(s[0], s[5]), s[10]), s[15]), s[20]);
        U64x C1 = xor64(xor64(xor64(xor64(s[1], s[6]), s[11]), s[16]), s[21]);
        U64x C2 = xor64(xor64(xor64(xor64(s[2], s[7]), s[12]), s[17]), s[22]);
        U64x C3 = xor64(xor64(xor64(xor64(s[3], s[8]), s[13]), s[18]), s[23]);
        U64x C4 = xor64(xor64(xor64(xor64(s[4], s[9]), s[14]), s[19]), s[24]);

        U64x D0 = xor64(C4, rotl64(C1, 1));
        U64x D1 = xor64(C0, rotl64(C2, 1));
        U64x D2 = xor64(C1, rotl64(C3, 1));
        U64x D3 = xor64(C2, rotl64(C4, 1));
        U64x D4 = xor64(C3, rotl64(C0, 1));

        U64x b00 = xor64(s[0], D0);
        U64x b10 = rotl64(xor64(s[1], D1), 1);
        U64x b20 = rotl64(xor64(s[2], D2), 62);
        U64x b05 = rotl64(xor64(s[3], D3), 28);
        U64x b15 = rotl64(xor64(s[4], D4), 27);
        U64x b16 = rotl64(xor64(s[5], D0), 36);
        U64x b01 = rotl64(xor64(s[6], D1), 44);
        U64x b11 = rotl64(xor64(s[7], D2), 6);
        U64x b21 = rotl64(xor64(s[8], D3), 55);
        U64x b06 = rotl64(xor64(s[9], D4), 20);
        U64x b07 = rotl64(xor64(s[10], D0), 3);
        U64x b17 = rotl64(xor64(s[11], D1), 10);
        U64x b02 = rotl64(xor64(s[12], D2), 43);
        U64x b12 = rotl64(xor64(s[13], D3), 25);
        U64x b22 = rotl64(xor64(s[14], D4), 39);
        U64x b23 = rotl64(xor64(s[15], D0), 41);
        U64x b08 = rotl64(xor64(s[16], D1), 45);
        U64x b18 = rotl64(xor64(s[17], D2), 15);
        U64x b03 = rotl64(xor64(s[18], D3), 21);
        U64x b13 = rotl64(xor64(s[19], D4), 8);
        U64x b14 = rotl64(xor64(s[20], D0), 18);
        U64x b24 = rotl64(xor64(s[21], D1), 2);
        U64x b09 = rotl64(xor64(s[22], D2), 61);
        U64x b19 = rotl64(xor64(s[23], D3), 56);
        U64x b04 = rotl64(xor64(s[24], D4), 14);

        s[0] = xor64(b00, andnot64(b01, b02));
        s[1] = xor64(b01, andnot64(b02, b03));
        s[2] = xor64(b02, andnot64(b03, b04));
        s[3] = xor64(b03, andnot64(b04, b00));
        s[4] = xor64(b04, andnot64(b00, b01));
        s[5] = xor64(b05, andnot64(b06, b07));
        s[6] = xor64(b06, andnot64(b07, b08));
        s[7] = xor64(b07, andnot64(b08, b09));
        s[8] = xor64(b08, andnot64(b09, b05));
        s[9] = xor64(b09, andnot64(b05, b06));
        s[10] = xor64(b10, andnot64(b11, b12));
        s[11] = xor64(b11, andnot64(b12, b13));
        s[12] = xor64(b12, andnot64(b13, b14));
        s[13] = xor64(b13, andnot64(b14, b10));
        s[14] = xor64(b14, andnot64(b10, b11));
        s[15] = xor64(b15, andnot64(b16, b17));
        s[16] = xor64(b16, andnot64(b17, b18));
        s[17] = xor64(b17, andnot64(b18, b19));
        s[18] = xor64(b18, andnot64(b19, b15));
        s[19] = xor64(b19, andnot64(b15, b16));
        s[20] = xor64(b20, andnot64(b21, b22));
        s[21] = xor64(b21, andnot64(b22, b23));
        s[22] = xor64(b22, andnot64(b23, b24));
        s[23] = xor64(b23, andnot64(b24, b20));
        s[24] = xor64(b24, andnot64(b20, b21));

        s[0] = xor64(s[0], KECCAK_RC[r]);
    }
}

__device__ __forceinline__ bool below_difficulty(const uint32_t h[8], const uint32_t d[8]) {
    for (int i = 0; i < 8; i++) {
        if (h[i] < d[i]) return true;
        if (h[i] > d[i]) return false;
    }
    return false;
}

__global__ void mine_kernel(
    const uint8_t* __restrict__ challenge,
    const uint32_t* __restrict__ difficulty,
    uint64_t base_nonce,
    uint32_t iters,
    Result* result
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = base_nonce + tid * (uint64_t)iters;

    #pragma unroll 1
    for (uint32_t k = 0; k < iters; k++, nonce++) {
        U64x s[25];
        for (int i = 0; i < 25; i++) s[i] = 0;

        s[0] = make_u64(load_le32(challenge + 0), load_le32(challenge + 4));
        s[1] = make_u64(load_le32(challenge + 8), load_le32(challenge + 12));
        s[2] = make_u64(load_le32(challenge + 16), load_le32(challenge + 20));
        s[3] = make_u64(load_le32(challenge + 24), load_le32(challenge + 28));
        s[7] = make_u64(bswap32d((uint32_t)(nonce >> 32)), bswap32d((uint32_t)nonce));
        s[8] = make_u64(0x00000001u, 0x00000000u);
        s[16] = make_u64(0x00000000u, 0x80000000u);

        keccak_f1600(s);

        uint32_t h[8];
        h[0] = bswap32d(s[0].lo);
        h[1] = bswap32d(s[0].hi);
        h[2] = bswap32d(s[1].lo);
        h[3] = bswap32d(s[1].hi);
        h[4] = bswap32d(s[2].lo);
        h[5] = bswap32d(s[2].hi);
        h[6] = bswap32d(s[3].lo);
        h[7] = bswap32d(s[3].hi);

        if (below_difficulty(h, difficulty)) {
            if (atomicCAS(&result->found, 0u, 1u) == 0u) {
                result->nonce = nonce;
                for (int i = 0; i < 8; i++) result->hash[i] = h[i];
            }
            return;
        }
    }
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::runtime_error("invalid hex character");
}

static std::string strip0x(std::string s) {
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) return s.substr(2);
    return s;
}

static void parse_hex32(const std::string& in, uint8_t out[32]) {
    std::string s = strip0x(in);
    if (s.size() != 64) throw std::runtime_error("expected 32-byte hex string");
    for (int i = 0; i < 32; i++) out[i] = (hexval(s[2 * i]) << 4) | hexval(s[2 * i + 1]);
}

static uint64_t parse_u64(const std::string& in) {
    std::string s = strip0x(in);
    if (s.size() != in.size()) return std::stoull(s, nullptr, 16);
    return std::stoull(s, nullptr, 10);
}

static void difficulty_words(const uint8_t bytes[32], uint32_t out[8]) {
    for (int i = 0; i < 8; i++) {
        out[i] = ((uint32_t)bytes[4 * i] << 24) | ((uint32_t)bytes[4 * i + 1] << 16) |
                 ((uint32_t)bytes[4 * i + 2] << 8) | (uint32_t)bytes[4 * i + 3];
    }
}

static std::vector<int> parse_gpus(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stoi(item));
    }
    return out;
}

static void print_hash(const uint32_t h[8]) {
    printf("0x");
    for (int i = 0; i < 8; i++) printf("%08x", h[i]);
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    bool have_challenge = false, have_difficulty = false, have_base = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (a == "--challenge") {
            parse_hex32(need("--challenge"), cfg.challenge);
            have_challenge = true;
        } else if (a == "--difficulty") {
            uint8_t b[32];
            parse_hex32(need("--difficulty"), b);
            difficulty_words(b, cfg.difficulty);
            have_difficulty = true;
        } else if (a == "--gpus") {
            cfg.gpus = parse_gpus(need("--gpus"));
        } else if (a == "--blocks") {
            cfg.blocks = (uint32_t)std::stoul(need("--blocks"));
        } else if (a == "--threads") {
            cfg.threads = (uint32_t)std::stoul(need("--threads"));
        } else if (a == "--iters") {
            cfg.iters = (uint32_t)std::stoul(need("--iters"));
        } else if (a == "--seconds") {
            cfg.seconds = std::stoi(need("--seconds"));
        } else if (a == "--base") {
            cfg.base_nonce = parse_u64(need("--base"));
            have_base = true;
        } else {
            throw std::runtime_error("unknown arg: " + a);
        }
    }

    if (!have_challenge || !have_difficulty) {
        throw std::runtime_error("usage: cuda_miner --challenge 0x... --difficulty 0x... [--gpus 0,1,2,3]");
    }
    if (cfg.gpus.empty()) {
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        for (int i = 0; i < count; i++) cfg.gpus.push_back(i);
    }
    if (!have_base) {
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        cfg.base_nonce = (uint64_t)now;
    }
    return cfg;
}

struct WorkerState {
    std::atomic<uint64_t> hashes{0};
    std::atomic<bool> done{false};
};

static void worker(const Config cfg, int gpu_index, int gpu_id, WorkerState* state, Result* host_result, std::atomic<bool>* stop) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    uint8_t* d_challenge = nullptr;
    uint32_t* d_difficulty = nullptr;
    Result* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_challenge, 32));
    CUDA_CHECK(cudaMalloc(&d_difficulty, 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(Result)));
    CUDA_CHECK(cudaMemcpy(d_challenge, cfg.challenge, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_difficulty, cfg.difficulty, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(Result)));

    uint64_t stride = (uint64_t)cfg.blocks * cfg.threads * cfg.iters;
    uint64_t base = cfg.base_nonce + (uint64_t)gpu_index * (1ULL << 56);
    Result local{};

    while (!stop->load(std::memory_order_relaxed)) {
        mine_kernel<<<cfg.blocks, cfg.threads>>>(d_challenge, d_difficulty, base, cfg.iters, d_result);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&local, d_result, sizeof(Result), cudaMemcpyDeviceToHost));
        state->hashes.fetch_add(stride, std::memory_order_relaxed);
        if (local.found) {
            *host_result = local;
            stop->store(true, std::memory_order_relaxed);
            break;
        }
        base += stride;
    }

    CUDA_CHECK(cudaFree(d_challenge));
    CUDA_CHECK(cudaFree(d_difficulty));
    CUDA_CHECK(cudaFree(d_result));
    state->done.store(true, std::memory_order_relaxed);
}

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        printf("Using %zu GPU(s), blocks=%u threads=%u iters=%u seconds=%d\n",
               cfg.gpus.size(), cfg.blocks, cfg.threads, cfg.iters, cfg.seconds);

        Result found{};
        std::atomic<bool> stop{false};
        std::vector<WorkerState> states(cfg.gpus.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < cfg.gpus.size(); i++) {
            threads.emplace_back(worker, cfg, (int)i, cfg.gpus[i], &states[i], &found, &stop);
        }

        auto started = std::chrono::steady_clock::now();
        uint64_t last_hashes = 0;
        auto last = started;
        while (!stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            uint64_t total = 0;
            for (auto& s : states) total += s.hashes.load(std::memory_order_relaxed);
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last).count();
            double elapsed = std::chrono::duration<double>(now - started).count();
            double rate = (double)(total - last_hashes) / dt;
            printf("hashrate %.2f GH/s, total %.3e hashes, elapsed %.0fs\n", rate / 1e9, (double)total, elapsed);
            fflush(stdout);
            last_hashes = total;
            last = now;
            if (elapsed >= cfg.seconds) stop.store(true, std::memory_order_relaxed);
        }

        for (auto& t : threads) t.join();

        if (found.found) {
            printf("FOUND nonce=0x%016llx hash=", (unsigned long long)found.nonce);
            print_hash(found.hash);
            printf("\n");
            return 0;
        }
        printf("No nonce found.\n");
        return 2;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
