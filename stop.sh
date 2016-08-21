PNAME=$1
echo "find process with name $PNAME"
PID=`ps ux | grep $PNAME | grep -v grep | awk '{print $2}'`

echo "kill $PID"
CMD="kill -9 $PID"
$CMD
