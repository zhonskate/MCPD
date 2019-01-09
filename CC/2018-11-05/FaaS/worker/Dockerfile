FROM node
ENV DOCKERVERSION=18.03.1-ce
RUN curl -fsSLO https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKERVERSION}.tgz \
  && tar xzvf docker-${DOCKERVERSION}.tgz --strip 1 \
  -C /usr/local/bin docker/docker \
  && rm docker-${DOCKERVERSION}.tgz
RUN mkdir /worker
WORKDIR /worker
RUN npm install --silent zeromq
COPY worker.js /worker/worker.js
RUN mkdir requestsworker
EXPOSE 2000
CMD ["node", "/worker/worker.js"]