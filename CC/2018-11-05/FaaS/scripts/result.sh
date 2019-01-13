for i in `seq $1 $2`
do
    printf "request $i ";curl http://localhost:3333/result/$i; echo "  "
done
