# Data sheet

**Dataset:** Fashion-MNIST, introduced by Han Xiao, Kashif Rasul and Roland Vollgraf (2017), from Zalando's article images. Official source: https://github.com/zalandoresearch/fashion-mnist. Upstream license: MIT.

**Contents:** 70,000 grayscale 28×28 garment images in ten categories: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. Original split: 60,000 training and 10,000 test images, balanced at 1,000 test images per class.

**Project split:** stratified 50,000 train and 10,000 validation images from the official training portion, fixed split seed 42 in the ten-seed comparison. Initialization/dropout/shuffle seeds are 0–9. The original baseline uses seed 42. The official test set is evaluated only after checkpoint selection; there was no test-driven architecture or hyperparameter search in the recorded study.

**Preprocessing:** decode verified IDX files; scale bytes to [0,1]; fit scalar mean and standard deviation on training pixels only; reuse them for validation and test. No image augmentation, synthetic image generation or relabeling is used.

**Integrity:** the downloader and every training/test read verify hardcoded SHA256 digests. Upstream URLs can move, but changed content is rejected. Original MD5 values are retained for attribution, not used as the sole integrity check. Split index hashes and normalization are recorded per run.

**Representativeness:** this is a curated, low-resolution garment dataset. It does not represent arbitrary camera images, current fashion inventories or unconstrained backgrounds. Visually overlapping labels can be difficult to separate at 28×28 resolution. Error inspection supplies hypotheses, not proven causal explanations.

**Distribution:** dataset files are downloaded from upstream and excluded from source/release archives. Figures display attributed dataset examples. No new data were collected from the user. Authoritative file hashes are in src/ml_lab/data.py; dataset provenance is in THIRD_PARTY.md.
