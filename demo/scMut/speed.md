```bash
run_test() {
    local method="$1"
    local transpose_flag="$2"
    local n_cells="$3"
    local n_sites="$4"
    local display_name="Test ${method} with train_transpose=${transpose_flag}"
    echo "[$(date '+%H:%M:%S')] $display_name started..."
    python << EOF && echo "[$(date '+%H:%M:%S')] $display_name over!"
import os,sys; sys.path.append(os.path.abspath("../../../"))
from VAETracer.scMut import test

test.run_pipe(
    run_model_method='$method',
    n_repeat=3,
    n_cells=$n_cells,
    n_sites=$n_sites,
    train_transpose=$transpose_flag,
    beta_pairs=[(1, 32, None, None)],
    model_params=dict(num_epochs=1000, num_epochs_nmf=1000, lr=1e-3, beta_kl=0.001, beta_best=0.001),
    train_params=dict(patience=45),
    load_params=dict(batch_size=1000, num_workers=0),
    cpu_time=False
)
EOF
}

# run test
run_test "vae+ft" False 10000 10000 | tee log_vae_3.txt
run_test "nmf+ft" False 10000 10000 | tee log_nmf_3.txt
run_test "vae+ft" False 1000 1000 | tee log_vae_2.txt
run_test "nmf+ft" False 1000 1000 | tee log_nmf_2.txt
run_test "vae+ft" False 100 100 | tee log_vae_1.txt
run_test "nmf+ft" False 100 100 | tee log_nmf_1.txt
```

nmf-0.1k
simple
nmf 140.90 131.04 137.94
ft 162.54 159.55 157.89
lineage
nmf 130.57 131.11 129.28
ft 156.11 154.03 162.46

nmf-1k
simple 
nmf 47.10 43.62 45.71
ft 53.01 53.66 49.68
lineage
nmf 43.69 43.82 40.42
ft 48.75 47.27 53.74

nmf-10k
simple 
nmf 1.69 1.69 1.70
ft 1.76 1.76 1.77
lineage
nmf 1.74 1.74 1.74
ft 1.81 1.81 1.82

vae-0.1k
simple 
vae-np 106.89 95.57 98.77
vae-xhat 253.95 221.26 257.60
ft 155.28 159.14 149.87
lineage
vae-np 112.79 116.27 116.87
vae-xhat 241.61 251.90 253.21
ft 160.80 153.87 162.75

vae-1k
simple 
vae-np 37.90 70.90 68.60
vae-xhat 98.48 94.29 98.32
ft 177.35 162.91 140.77
lineage
vae-np 65.31 63.66 64.38
vae-xhat 97.17 96.30 98.67
ft 183.26 177.84 160.09

vae-10k
simple 
vae-np 1.57 1.62 1.61
vae-xhat 2.35 2.31 2.36
ft 1.92 1.87 1.91
lineage
vae-np 1.62 1.62 1.62
vae-xhat 2.76 2.76 2.77
ft 1.91 1.91 1.91