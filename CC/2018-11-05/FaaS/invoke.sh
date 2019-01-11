max=100
for i in `seq 2 $max`
do
    curl --header "Content-Type: application/json" --request PUT --data '{"a":10,"b":2}' http://localhost:3333/invokefunction/8e4309e150a1e3ef8b4857851f3c6435f209ca7f4f6946d82739e1b24c945bbb
done
