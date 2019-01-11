max=100
for i in `seq 2 $max`
do
    curl http://localhost:3333/result/$i
done
