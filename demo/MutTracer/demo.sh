#!/bin/bash
VAETRACER_DIR=$(readlink -m ../..)
OUTPUT_DIR="${VAETRACER_DIR}/demo/MutTracer/output_test"
mkdir -p $OUTPUT_DIR && cd $VAETRACER_DIR

echo "Running MutTracer test..."

PYTHONPATH=${VAETRACER_DIR}/.. python -m MutTracer.main \
  --scmut_model_path ${VAETRACER_DIR}/demo/MutTracer/input_test/zmt_model_test.pkl \
  --zmt_path ${VAETRACER_DIR}/demo/MutTracer/input_test/z_mt_test.pt \
  --zxt_path ${VAETRACER_DIR}/demo/MutTracer/input_test/z_xt_test.pt \
  --input_times 1 2 \
  --predict_times 0 \
  --epochs 20 \
  --adata_path ${VAETRACER_DIR}/demo/MutTracer/input_test/exp.h5ad \
  --scvi_model_path ${VAETRACER_DIR}/demo/MutTracer/input_test/exp_scvi_model.pkl \
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
