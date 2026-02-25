#!/bin/bash
VAETRACER_DIR=$(readlink -m ../..)
export PYTHONPATH="${VAETRACER_DIR}/..:$PYTHONPATH"

OUTPUT_DIR="${VAETRACER_DIR}/demo/MutTracer/output_test"
SCVI_MODEL_PATH="${VAETRACER_DIR}/demo/MutTracer/input_test/exp_scvi_model.pkl"

mkdir -p $OUTPUT_DIR && cd "$VAETRACER_DIR"


echo "Pre-check..."
# Test if the scVI model file exists and can be loaded with pickle
# anndata package version difference may cause pickle-load error

test_scvi_model() {
    local path=$1
    PKL_PATH="$path" python -c "
import os, pickle
import scvi
pkl_path = os.environ['PKL_PATH']
try:
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
except:
    exit(1)
" >/dev/null 2>&1
}
regenerate_scvi_model() {
    local model_dir=$(dirname "$SCVI_MODEL_PATH")/linear_scvi
    local backup_path="${SCVI_MODEL_PATH}.bak"
    SCVI_MODEL_DIR="$model_dir" \
    SCVI_MODEL_BAK="$backup_path" \
    python -c "
import os, pickle
from scvi.model import LinearSCVI
model_dir = os.environ['SCVI_MODEL_DIR']
backup_path = os.environ['SCVI_MODEL_BAK']
try:
    model = LinearSCVI.load(model_dir)
    with open(backup_path, 'wb') as f:
        pickle.dump(model, f)
except:
    exit(1)
" >/dev/null 2>&1
}

test_scvi_model "$SCVI_MODEL_PATH" || {
    echo "Failed to load scVI model from $SCVI_MODEL_PATH"
    echo "Attempting re-generate pickle from model_path... "

    regenerate_scvi_model && {
        test_scvi_model "${SCVI_MODEL_PATH}.bak" && SCVI_MODEL_PATH="${SCVI_MODEL_PATH}.bak"
    } || {
        echo " MutTracer installation test FAILED "
        exit 1
    }
}


echo "Running MutTracer test..."

python -m MutTracer.main \
  --scmut_model_path ${VAETRACER_DIR}/demo/MutTracer/input_test/zmt_model_test.pkl \
  --zmt_path ${VAETRACER_DIR}/demo/MutTracer/input_test/z_mt_test.pt \
  --zxt_path ${VAETRACER_DIR}/demo/MutTracer/input_test/z_xt_test.pt \
  --input_times 1 2 \
  --predict_times 0 \
  --epochs 20 \
  --adata_path ${VAETRACER_DIR}/demo/MutTracer/input_test/linear_scvi/adata.h5ad \
  --scvi_model_path ${SCVI_MODEL_PATH} \
  --save_dir $OUTPUT_DIR \
  --real_times_keep 1 2  --pred_times_keep 1 \
  > ${OUTPUT_DIR}/demo.log 2>&1


echo "Checking output..."

PKL_COUNT=$(ls $OUTPUT_DIR/*.pkl 2>/dev/null | wc -l)

if [ "$PKL_COUNT" -gt 0 ]; then
    echo " MutTracer installation test SUCCESS "
else
    echo " MutTracer installation test FAILED "
fi
