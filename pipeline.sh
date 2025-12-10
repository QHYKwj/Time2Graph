#!/bin/bash

TARGET_FILE="scripts/cache/风场1/m01/0.embeddings"

# 设置错误时退出
set -e

# ============================
# 配置参数区域（可根据需要修改）
# ============================
K=15
C=300
NUM_SEGMENT=4
SEG_LENGTH=5
EMBED="concate"
TRANSFORMER_HEADS=4
TRANSFORMER_LAYERS=2
LR=0.001
PERCENTILE=10
FARM="风场1"
TURBINE_ID="m01"
NUM_EPOCHS=3
REPRESENTATION_SIZE=256

# ============================
# 路径配置
# ============================
TIME2GRAPH_DIR="time2graph/Time2Graph"
DEEPWALK_DIR="weighted_deepwalk"
SCRIPT_PATH="scripts/run2.py"
CACHE_DIR="scripts/cache/${FARM}/${TURBINE_ID}"
EDGELIST_FILE="${CACHE_DIR}/0.edgelist"
EMBEDDINGS_FILE="${CACHE_DIR}/0.embeddings"

echo "=========================================="
echo "开始运行 Time2Graph + DeepWalk 流水线"
echo "=========================================="
echo "风场: ${FARM}"
echo "风机ID: ${TURBINE_ID}"
echo "=========================================="

# ============================
# 第一步：运行 time2graph 生成 edgelist
# ============================
echo ""
echo "[步骤 1/3] 运行 time2graph 生成 .edgelist 文件..."
echo "激活 time2graph 环境..."

# conda init
# conda deactivate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Time2Graph

echo "执行 time2graph (第一次)..."
python ${SCRIPT_PATH} \
  --K ${K} \
  --C ${C} \
  --num_segment ${NUM_SEGMENT} \
  --seg_length ${SEG_LENGTH} \
  --embed ${EMBED} \
  --transformer_heads ${TRANSFORMER_HEADS} \
  --transformer_layers ${TRANSFORMER_LAYERS} \
  --lr ${LR} \
  --percentile ${PERCENTILE} \
  --farm ${FARM} \
  --turbine_id ${TURBINE_ID} \
  --num_epochs ${NUM_EPOCHS} || echo "第一次运行预期会因缺少 embeddings 文件而终止"

# 检查 edgelist 文件是否生成
if [ ! -f "${EDGELIST_FILE}" ]; then
    echo "错误：.edgelist 文件未生成！"
    exit 1
fi
echo "✓ .edgelist 文件生成成功"

if [ -f "$TARGET_FILE" ]; then
    echo "文件 $TARGET_FILE 已存在，只执行步骤一。"
    exit 0
fi

# ============================
# 第二步：运行 deepwalk 生成 embeddings
# ============================
echo ""
echo "[步骤 2/3] 运行 deepwalk 生成 .embeddings 文件..."
echo "切换到 deepwalk 目录..."

cd ../../${DEEPWALK_DIR}
conda deactivate
conda activate deepwalk

echo "执行 deepwalk..."
deepwalk \
  --input ../${TIME2GRAPH_DIR}/${EDGELIST_FILE} \
  --format weighted_edgelist \
  --output ../${TIME2GRAPH_DIR}/${EMBEDDINGS_FILE} \
  --representation-size ${REPRESENTATION_SIZE}

# 检查 embeddings 文件是否生成
if [ ! -f "../${TIME2GRAPH_DIR}/${EMBEDDINGS_FILE}" ]; then
    echo "错误：.embeddings 文件未生成！"
    exit 1
fi
echo "✓ .embeddings 文件生成成功"

# ============================
# 第三步：再次运行 time2graph 完成训练
# ============================
echo ""
echo "[步骤 3/3] 运行 time2graph 完成完整训练..."
echo "切换回 time2graph 目录..."

cd ../${TIME2GRAPH_DIR}
conda deactivate
conda activate Time2Graph

echo "执行 time2graph (第二次，完整运行)..."
python ${SCRIPT_PATH} \
  --K ${K} \
  --C ${C} \
  --num_segment ${NUM_SEGMENT} \
  --seg_length ${SEG_LENGTH} \
  --embed ${EMBED} \
  --transformer_heads ${TRANSFORMER_HEADS} \
  --transformer_layers ${TRANSFORMER_LAYERS} \
  --lr ${LR} \
  --percentile ${PERCENTILE} \
  --farm ${FARM} \
  --turbine_id ${TURBINE_ID} \
  --num_epochs ${NUM_EPOCHS}

echo ""
echo "=========================================="
echo "✓ 流水线执行完成！"
echo "=========================================="
echo "生成的文件位置："
echo "  - EdgeList: ${EDGELIST_FILE}"
echo "  - Embeddings: ${EMBEDDINGS_FILE}"
echo "=========================================="