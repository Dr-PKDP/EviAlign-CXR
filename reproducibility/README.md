# Reproducibility notes

`table_to_run.csv` maps each table of the paper to the runs and evaluation files it is computed from and to the script or tool that produces it.
`environment.txt` lists the software versions of the reported runs. Training configurations are in `configs/` (`main.yaml`, `ablations/`); the seeds
of the replicated systems are in `scripts/03_train_seeds.sh`. The fixed evaluation subsets are the first 100 batches of 32 images of each test list
(`--max_batches 100`, `configs/eval_faith.yaml`), in manifest order, so they are deterministic given `manifests/`. Random-removal masks are seeded by
image, finding and replicate (`--rand_key`), so a shared pair receives the same control in every model.
