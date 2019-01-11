max=$1
for i in `seq 2 $max`
do
    curl --header "Content-Type: application/json" --request PUT --data '{"num_dices":10,"size":20}' http://localhost:3333/invokefunction/778ef69f1a904a46c737f60f688b8cd867e89f49fcaceffe213587f470b467fe; echo "  "
done
