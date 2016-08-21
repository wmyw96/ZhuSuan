G1="192.168.245.99"
G2="192.168.245.100"
G3="192.168.245.153"
G4="192.168.245.154"
G5="192.168.245.155"
G6="192.168.245.156"

HOME="/home/shizhen"
PROG="$HOME/ZhuSuan/examples/vae.py"
LOG_PATH="$HOME/ZhuSuan/logs"
#WORKERS=2

if [ "$#" -ne "2" ]; then
    echo "Usage: $0 <JOB_NAME> <TASK_INDEX>"
    exit -1
fi
PS_PORT=2222
WORKER_PORT=3333
JOB_NAME=$1
TASK_INDEX=$2

PS_URL="$G1"
PS_HOSTS=${PS_URL}":"${PS_PORT}

WORKER_URL="$G5 $G6"
#WORKER_URL="$G5"
NUM_WORKERS=0

for i in $WORKER_URL
do
    if [ "$WORKER_HOSTS" == "" ]; then
        WORKER_HOSTS="${i}:${WORKER_PORT}"
    else
        WORKER_HOSTS="${WORKER_HOSTS},${i}:${WORKER_PORT}"
    fi
    ((NUM_WORKERS++))
done



CMD="python $PROG 
	--ps_hosts=$PS_HOSTS
	--worker_hosts=$WORKER_HOSTS
	--job_name=$JOB_NAME 
	--task_index=$TASK_INDEX
	--num_workers=$NUM_WORKERS"


echo $CMD
echo "Log will save to $LOG_PATH/$JOB_NAME-$TASK_INDEX.log"
$CMD 2>&1 | tee $LOG_PATH/$JOB_NAME-$TASK_INDEX.log
