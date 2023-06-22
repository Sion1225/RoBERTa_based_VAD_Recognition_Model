# Pre-trained Correlation Models & Fine-tuned RoBERTa Models for VAD Assinging

## Correlation Models

### FFNN

__He_normal__
|Version|   MSE   |Units|Batch size|Epochs|Dropout rate|L2 (activity)|L2 (kernel) |
|-------|---------|-----|----------|------|------------|-------------|------------|
| Ver.1 |0.0004832| 429 |    18    |  12  |0.0141742883|0.00067094843|0.000216125 |
| Ver.2 |0.0001023| 475 |    43    |  14  |0.1342275249|0.00222985367|0.0015969552|
| Ver.3 |0.000483 | 237 |    12    |  13  |0.2635720489|0.00278011647|0.000919571 |

__He_uniform__
|Version|   MSE   |Units|Batch size|Epochs|Dropout rate|L2 (activity)|L2 (kernel) |
|-------|---------|-----|----------|------|------------|-------------|------------|
| Ver.4 |0.0048504| 208 |     8    |   8  |0.1031961660|0.00237429775|0.000753898 |
| Ver.5 |0.0022572| 477 |    21    |  13  |0.2456046038|0.00117673684|0.0003747580|
| Ver.6 |0.0005390| 546 |    30    |  15  |0.0553479118|0.00089716754|0.0005002549|

## Fine-tuned RoBERTa Models for VAD Assinging
Fine-tuned RoBERTa Models for VAD Assinging too big to upload on github so I'm considering upload it on huggingface...
