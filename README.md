# CUDA HASH miner benchmark

Standalone CUDA miner for testing remote NVIDIA rigs. It mines against a supplied
challenge and difficulty, prints aggregate hashrate, and prints a nonce/hash when
one is found.

Build:

```bash
cd cuda
make CUDA_ARCH=sm_89
```

Run:

```bash
./cuda_miner \
  --challenge 0x7815f1ac49c179212a365a5ae9bdd2d9b19c7cc643fbaea73f88c43cea648af4 \
  --difficulty 0x000000000000ffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --seconds 60
```

Options:

```text
--gpus 0,1,2,3       GPU ids to use. Default: all visible GPUs.
--blocks N           Blocks per GPU launch. Default: 131072.
--threads N          Threads per block. Default: 256.
--iters N            Nonces per CUDA thread. Default: 128.
--seconds N          Stop after N seconds unless a nonce is found. Default: 60.
--base HEX_OR_DEC    Starting nonce. Default: random-ish time seed.
```

For RTX 4090 use `CUDA_ARCH=sm_89`.
