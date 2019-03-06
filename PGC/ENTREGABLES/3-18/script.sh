docker network create --subnet=172.20.0.0/24 red1
docker network create --subnet=172.21.0.0/24 red2
docker run -d --name proxy -p 2367:22 --network=red1 ajoergensen/openssh-server
docker run -dit --name web --network=red1 nginx
docker run -dit --name server --network=red1 alpine
docker network connect red2 server
docker run -dit --name db --network=red2 alpine
docker exec -it proxy adduser pgcuser
