TOTAL_TASKS=100
BATCH_SIZE=100

if [ $# != 5 ] && [ $# != 6 ]; then
    echo "Error: 5 or 6 arguments required."
    exit 1
fi

CONFIG_FILE=$1
RESULT_PATH=$2
NODE_ALL=$3
NODE_THIS=$4
START_IDX=$5
SPLIT_FILE=$6

if [ -z "$SPLIT_FILE" ]; then
    SPLIT_SUFFIX=""
else
    SPLIT_SUFFIX="-s $SPLIT_FILE"
fi

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))
    if [ $NODE_TARGET == $NODE_THIS ]; then
        echo "Task ${i} assigned to this worker (${NODE_THIS})"
        echo "+ python -m scripts.sample_diffusion ${CONFIG_FILE} -i ${i} --batch_size ${BATCH_SIZE} --result_path ${RESULT_PATH} ${SPLIT_SUFFIX}"
        python -m scripts.sample_diffusion ${CONFIG_FILE} -i ${i} --batch_size ${BATCH_SIZE} --result_path ${RESULT_PATH} ${SPLIT_SUFFIX}
    fi
done
