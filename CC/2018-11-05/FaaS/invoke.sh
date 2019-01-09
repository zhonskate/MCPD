max=100
for i in `seq 2 $max`
do
    curl --header "Content-Type: application/json" --request PUT --data '{"a":10,"b":2}' http://localhost:3333/invokefunction/3266a184aeec777fac39d6e7b13cadeec71d4b894a4d4707191f2458b8b2af84
done
