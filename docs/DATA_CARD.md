# BANKING77 data card

## Source and attribution

BANKING77 by PolyAI: 10,003 training examples, 3,080 test examples and 77 banking intents. The dataset is English and contains online-banking customer queries.

Source: https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data

License: Creative Commons Attribution 4.0 International. Copyright and attribution remain with PolyAI and the dataset authors. See the [upstream LICENSE](https://github.com/PolyAI-LDN/task-specific-datasets/blob/57ec275d8078af65b7731c2a98be812d844a6d6b/LICENSE) and https://creativecommons.org/licenses/by/4.0/.

Citation: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan Vulić. *Efficient Intent Detection with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for Conversational AI, 2020. https://arxiv.org/abs/2003.04807

## Integrity

| File | SHA-256 |
| --- | --- |
| `train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| `test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` |

The files are downloaded verbatim to ignored `data/raw/`. Changes for the experiment consist of train-only filtering and train partitioning; no original file is overwritten. Exact whitespace/case-normalized training duplicates are removed, and training rows whose normalized text occurs in official test are excluded. This leaves 9,992 training rows. The official test remains unchanged, including its repeated text; a secondary report deduplicates it.

The fixed 60/20/20 development split is stratified by class. The complete test label distribution is not used for model or threshold choices. Unit-test fixtures are small authored examples, separate from the benchmark and not included in reported scores.

## Limitations

The benchmark is constrained to banking intents. No claim about passenger messages, Greek, arbitrary topics, real-time bank systems or current products is supported. Text duplicates are only one kind of contamination; semantic near-duplicates and dataset annotation issues may remain. Public benchmark performance can be affected by repeated prior experimentation on the same official test.
