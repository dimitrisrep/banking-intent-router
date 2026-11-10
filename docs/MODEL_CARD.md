# Banking Intent Router model card

Intended use: a reproducible portfolio experiment and local intent-routing demonstration.

The candidate pipelines use word TF-IDF (unigrams/bigrams) and character-within-word TF-IDF (3-5 grams), with separate vocabulary limits of 40,000 features. Candidate classifiers are a linear SVM and an SGD logistic classifier. Both are calibrated with held-out sigmoid calibration while the base pipeline is frozen. Classifier and threshold selection use a distinct validation partition.

In the measured experiment, validation macro F1 was 0.8932 for the linear SVM and 0.8979 for SGD logistic. SGD logistic was selected before reading final test results. On official test it achieved macro F1 0.8921 and accuracy 0.8919. Scores sum across the 77 in-domain classes; that does not establish that the input belongs to the supported domain.

The review threshold maximizes validation coverage subject to >=90% accepted accuracy and >=100 accepted examples. Test accepted accuracy was 0.8950, below the target. Expected calibration error was 0.1317. Neither calibration nor abstention should be described as a guarantee.

The model is not tested for out-of-domain detection, prompt attacks, fairness, live distribution shift, adversarial spelling or non-English requests. There is no external bank connection and no action execution. The API has no authentication or rate limiting and is intended to be run on localhost. Use a gateway and appropriate controls before any network deployment.

Runtime: CPU, Python 3.12, scikit-learn 1.8.0 in the recorded run. Software versions, split sizes, hashes and latency scope are in `reports/metrics.json`. The raw corpus and artifacts are reproduced by the CLI. The implementation and documentation were developed with AI assistance.
