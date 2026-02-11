#!/bin/bash
run_test() {
    local method="$1"
    local transpose_flag="$2"
    local display_name="Test ${method} with train_transpose=${transpose_flag}"
    echo "[$(date '+%H:%M:%S')] $display_name started..."
    python << EOF && echo "[$(date '+%H:%M:%S')] $display_name over!"
# import scMut
import os,sys; sys.path.append(os.path.abspath("../../../"))
from VAETracer.scMut import test, save_model_to_pickle, save_model_to_adata

# run test
final_result = test.run_pipe(
    run_model_method='$method',
    n_repeat=1,
    n_cells=100,
    n_sites=100,
    train_transpose=$transpose_flag,
    beta_pairs=[(1, 32, None, None)],
    model_params=dict(num_epochs=1000, num_epochs_nmf=1000, lr=1e-3, beta_kl=0.001, beta_best=0.001),
    train_params=dict(patience=45),
    load_params=dict(batch_size=5000, num_workers=0),
    cpu_time=False,
    return_model=True
)

# extract model
simple_model = final_result['simple'][0][1]['model']
lineage_model = final_result['lineage'][0][1]['model']

# save model into pickle file
save_model_to_pickle(simple_model, 'simple_model.pkl')
save_model_to_pickle(lineage_model, 'lineage_model.pkl')

# save model-results into anndata-h5ad file
save_model_to_adata(simple_model).write_h5ad('simple_model_result.h5ad')
save_model_to_adata(lineage_model).write_h5ad('lineage_model_result.h5ad')
EOF
}

# run test
run_test "nmf+vae+ft" False
