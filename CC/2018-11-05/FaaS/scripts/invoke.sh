max=$(($3 + 1))

for i in `seq 2 $max`
do
    curl --header "Content-Type: application/json" --request PUT --data $2 http://localhost:3333/invokefunction/$1; echo "  "
    if [ ! -z "$4" ]
    then
        sleep $4
    fi
done
