Replacement for the two miles bullets in our comment on #4272 (https://github.com/radixark/miles/pull/4272#issuecomment-6531805913). The `-r3` compare link shows an empty diff since both changes moved onto one branch. Not posted.

```markdown
- [Session server on token space](https://github.com/radixark/miles/compare/rockdu/vlm-02-token-space...dzorlu:miles:tito-session-token-space), one branch with two commits:
  1. Each turn sends the stored expanded prefix, the new part with one placeholder per image, and `mm_token_expansion_start_len`. The session stores the engine's returned expanded IDs, and the trainer uses your `compute_multimodal_train_inputs`.
  2. [R3 offsets](https://github.com/dzorlu/miles/commit/dce628a206): routed-expert rows are counted on the expanded tokens, so they stay aligned across turns. This only matters once screenshots go through the session server, so both go in one PR.
```
