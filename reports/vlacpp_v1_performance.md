# vlacpp v1 LIBERO Performance

This report records the pi0 LIBERO comparison used for the v1 README.

## Checkpoint

- Model: `lerobot/pi0_libero_finetuned_v044`
- Local checkpoint used during measurement:
  `ckpts/pepijn-pi0-libero-finetuned-extra2-v044-local`
- LeRobot environment: `lerobot==0.4.4`, `transformers==4.53.3`,
  `torch==2.10.0+cu128`
- OpenPI environment: checkout at commit `c23745b`, Python 3.11.5,
  `jax==0.5.3`, `jaxlib==0.5.3`, `torch==2.7.1`, `libero==0.1.0`,
  `robosuite==1.4.1`, `mujoco==2.3.7`
- Hardware: Intel Xeon Platinum 8358P CPU, NVIDIA A100-PCIE-40GB GPU

## Reading

vlacpp v1 is faster than LeRobot CUDA with `compile_model=False`, but slower
than checkpoint-compiled LeRobot CUDA. On CPU, vlacpp is much faster on measured
chunks: about `5.9 s` per action chunk versus about `14 s` for LeRobot native
CPU.

## Results

CUDA rows use warm chunk timing, excluding the first compile/warmup chunk.

![vlacpp v1 benchmark comparison](vlacpp_v1_benchmark.svg)
