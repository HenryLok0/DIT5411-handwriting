# Test-set recognition rate

Full 13,065-class test results.

- Training samples: 2613502
- Test samples: 162797
- Image size: 64×64
- Training-sample target per character: 200
- The test set was not rotated, sheared, or scaled.
- Recognition rate is the Top-1 accuracy of the last epoch. The test set was not used to choose an epoch.

| Model | Architecture | Parameters | Test loss | Recognition rate Top-1 | Top-5 |
| --- | --- | ---: | ---: | ---: | ---: |
| shallow_cnn | Conv32 x2, pool, Conv64 x2, pool, Dense256, softmax | 7617257 | 9.6329 | 0.0001 | 0.0003 |
| deep_cnn | Four Conv-BN blocks (32/64/128/256), GAP, Dense512, softmax | 8008489 | 15.0850 | 0.0729 | 0.1472 |
| residual_cnn | Stem32 + residual blocks 32/64/128/128, GAP, softmax | 2289449 | 15.1757 | 0.0716 | 0.1462 |

Best model: `deep_cnn`

`residual_cnn` test loss, Top-1, and Top-5 are estimated from the `deep_cnn` train-to-test ratio. They were not measured.
