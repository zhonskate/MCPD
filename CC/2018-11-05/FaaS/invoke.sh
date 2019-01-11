max=$1
for i in `seq 2 $max`
do
    curl --header "Content-Type: application/json" --request PUT --data '{"a":10,"b":2}' http://localhost:3333/invokefunction/520302ba0b8854f03595629c574e2a4f5dda719168a90d910d483318f5f7a76e; echo "  "
done
