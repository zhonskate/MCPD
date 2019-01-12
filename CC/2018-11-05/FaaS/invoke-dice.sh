max=$1
for i in `seq 2 $max`
do
    curl --header "Content-Type: application/json" --request PUT --data '{"num_dices":1,"size":100}' http://localhost:3333/invokefunction/510f4b4b94bf55b78ac277d6c397781a0a39844386c7d0938c8cb977b396ed79; echo "  "
done
