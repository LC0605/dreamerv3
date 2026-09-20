# Stage S3-A A1-Recovery

- Parent: `stageS3A_Retry_A1_4000/ckpt/20260830T121605F166990`
- Frozen anchor reference: `stageS2_Conservative_6000/ckpt/20260828T154503F849521`
- Updates: 4000
- Episode sampling distribution: 40% `[+0.30,+0.45]`, 30%
  `[+0.45,+0.60]`, 20% `[+0.60,+0.80]`, 10% fixed +0.3 anchored replay.
- Low-range random histories and fixed +0.3 replay receive Conservative-2k
  near-obstacle anchor targets. Middle/high random histories remain unanchored.
- Anchor scale, learning rates, reward, observations, actions, Dreamer losses,
  `bc_scale=0`, and PPO-disabled settings are unchanged from A1.
