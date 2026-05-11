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

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ __forceinline__ uint32_t bswap32d(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint64_t bswap64d(uint64_t x) {
    return (uint64_t)bswap32d((uint32_t)(x >> 32)) | ((uint64_t)bswap32d((uint32_t)x) << 32);
}

__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ __forceinline__ uint64_t load_le64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)p[i]) << (8 * i);
    return v;
}

__device__ void keccak_f1600(uint64_t s[25]) {
    for (int r = 0; r < 24; r++) {
        uint64_t C[5], D[5], B[25];
        for (int x = 0; x < 5; x++) {
            C[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
        }
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                s[x + 5 * y] ^= D[x];
            }
        }

        B[0]  = s[0];
        B[10] = rotl64(s[1], 1);
        B[20] = rotl64(s[2], 62);
        B[5]  = rotl64(s[3], 28);
        B[15] = rotl64(s[4], 27);
        B[16] = rotl64(s[5], 36);
        B[1]  = rotl64(s[6], 44);
        B[11] = rotl64(s[7], 6);
        B[21] = rotl64(s[8], 55);
        B[6]  = rotl64(s[9], 20);
        B[7]  = rotl64(s[10], 3);
        B[17] = rotl64(s[11], 10);
        B[2]  = rotl64(s[12], 43);
        B[12] = rotl64(s[13], 25);
        B[22] = rotl64(s[14], 39);
        B[23] = rotl64(s[15], 41);
        B[8]  = rotl64(s[16], 45);
        B[18] = rotl64(s[17], 15);
        B[3]  = rotl64(s[18], 21);
        B[13] = rotl64(s[19], 8);
        B[14] = rotl64(s[20], 18);
        B[24] = rotl64(s[21], 2);
        B[9]  = rotl64(s[22], 61);
        B[19] = rotl64(s[23], 56);
        B[4]  = rotl64(s[24], 14);

        for (int y = 0; y < 5; y++) {
            int o = 5 * y;
            for (int x = 0; x < 5; x++) {
                s[o + x] = B[o + x] ^ ((~B[o + ((x + 1) % 5)]) & B[o + ((x + 2) % 5)]);
            }
        }
        s[0] ^= KECCAK_RC[r];
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
        uint64_t s[25];
        for (int i = 0; i < 25; i++) s[i] = 0;

        s[0] = load_le64(challenge + 0);
        s[1] = load_le64(challenge + 8);
        s[2] = load_le64(challenge + 16);
        s[3] = load_le64(challenge + 24);
        s[7] = bswap64d(nonce);
        s[8] = 0x0000000000000001ULL;
        s[16] = 0x8000000000000000ULL;

        keccak_f1600(s);

        uint32_t h[8];
        h[0] = bswap32d((uint32_t)s[0]);
        h[1] = bswap32d((uint32_t)(s[0] >> 32));
        h[2] = bswap32d((uint32_t)s[1]);
        h[3] = bswap32d((uint32_t)(s[1] >> 32));
        h[4] = bswap32d((uint32_t)s[2]);
        h[5] = bswap32d((uint32_t)(s[2] >> 32));
        h[6] = bswap32d((uint32_t)s[3]);
        h[7] = bswap32d((uint32_t)(s[3] >> 32));

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
