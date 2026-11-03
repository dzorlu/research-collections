# Which model wrote this token?

In RL for language models, the model that writes the tokens is usually not the model being trained. Every correction
in this post follows from one rule:

**Train as if the model you are updating right now had written the batch.**

The weight that does this, for each token, is

```python
w = p_now / p_engine
```

`p_now` is the probability of the token under the model you are updating. `p_engine` is its probability under the
model that actually wrote it.

## 1. Why the weight works

Two tokens, "cat" and "dog". The engine wrote 10 samples and picked "cat" 10% of the time:

```python
batch = ["cat", "dog", "dog", "dog", "dog", "dog", "dog", "dog", "dog", "dog"]   # 1 cat, 9 dogs
```

The model you are updating picks "cat" 20% of the time. Had it written the batch, there would be 2 cats and 8 dogs.

Your batch is missing a cat. So the one cat you have counts twice, and each dog counts a bit less:

```python
w_cat = 0.2 / 0.1 = 2.0
w_dog = 0.8 / 0.9 = 0.89

1 cat  * 2.0  = 2 cats
9 dogs * 0.89 = 8 dogs     # the batch now counts like the updated model wrote it
```

Table 1: weights for the batch above.

| token | engine wrote it | updated model would write it | weight |
|---|---|---|---|
| cat | 10% | 20% | 2.0 |
| dog | 90% | 80% | 0.89 |

The weight does not say "cat" is a better token. Whether a token is good comes from the advantage, not the weight.
The weight only corrects how often each token shows up.

## 2. Why the two models differ

Three reasons, all common in practice:

1. **Lag.** The engine samples while the trainer updates, so the engine holds weights from a few updates ago.
2. **Numerics.** The engine and the trainer run different kernels and precisions, so even the same weights give
   slightly different probabilities.
3. **Minibatches.** One batch is split into several minibatches with an optimizer step after each. By the last
   minibatch, the model has moved several times since the batch was written.

## 3. Two legs

Reasons 1 and 2 happen before training starts. Reason 3 happens during it. So the trip from the engine to the model
you are updating has two legs, with a stop in between:

```python
engine  --(p_old / p_engine)-->  old  --(p_now / p_old)-->  now
           mismatch: lag + numerics       staleness: minibatch steps
```

`old` is the trainer at the start of the batch, before any minibatch step. It never wrote anything. It is only a
stop on the way, computed by running the trainer once over the batch.

The two weights multiply, and `p_old` cancels:

```python
(p_now / p_old) * (p_old / p_engine) = p_now / p_engine
```

The target is still the model you are updating right now.

## 4. Why split it

Each leg gets its own limit.

Table 2: which leg each method limits.

| method | staleness `p_now / p_old` | mismatch `p_old / p_engine` |
|---|---|---|
| PPO | clipped | ignored |
| PPO + TIS | clipped | capped at 2 |
| DPPO | one limit on `p_now / p_engine`, no split | |

With one optimizer step per batch, `p_now = p_old`, so the staleness leg is exactly 1 and only mismatch is left.
