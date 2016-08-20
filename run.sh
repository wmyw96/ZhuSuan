G2="192.168.245.100:2333"
G3="192.168.245.153:2333"
G4="192.168.245.154:2333"
G5="192.168.245.155:2333"
G6="192.168.245.156:2333"

USER="/home/yama"
PROG="$USER/mfs/tf-dist/caicloud/mnist_cnn.py"
PROG="$USER/mfs/tf-dist/mnist/mnist_replica.py"
PROG="$USER/mfs/ZhuSuan-inter-node/examples/vae.py"

HOST=`hostname`

if [[ $HOST == "jungpu3" ]]
then
	JOB_NAME="ps"
	TASK_INDEX=0
elif [[ $HOST == "jungpu5" ]]
then
	JOB_NAME="worker"
	TASK_INDEX=0
elif [[ $HOST == "jungpu6" ]]
then
	JOB_NAME="worker"
	TASK_INDEX=1
else
	JOB_NAME="invalid"
	TASK_INDEX=-1
	echo "$HOST was not supposed to run $PROG"
fi

LOG_PATH="$USER/mfs/logs"

WORKER_HOSTS="$G5,$G6"
WORKER_GRPC_URL="grpc://$G5,grpc://$G6"
WORKER_HOSTS="$G5"
WORKER_GRPC_URL="grpc://$G5"

CMD="python $PROG 
	--ps_hosts=$G3
	--worker_hosts=$WORKER_HOSTS
	--worker_grpc_url=$WORKER_GRPC_URL
	--job_name=$JOB_NAME 
	--task_index=$TASK_INDEX"

echo $CMD
echo "Log will save to $LOG_PATH/$JOB_NAME-$HOST.log"
$CMD 2>&1 | tee $LOG_PATH/$JOB_NAME-$HOST.log
