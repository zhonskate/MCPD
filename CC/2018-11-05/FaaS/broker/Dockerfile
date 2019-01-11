FROM node
ENV DOCKERVERSION=18.03.1-ce
RUN curl -fsSLO https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKERVERSION}.tgz \
  && tar xzvf docker-${DOCKERVERSION}.tgz --strip 1 \
  -C /usr/local/bin docker/docker \
  && rm docker-${DOCKERVERSION}.tgz 
RUN mkdir /server
WORKDIR /server
RUN mkdir requests
RUN npm install --silent express && \
 npm install --silent collections  && \
 npm install --silent queue-fifo && \
 npm install --silent multer && \
 npm install --silent cors && \
 npm install --silent lokijs && \
 npm install --silent del && \
 npm install --silent js-sha256 && \
 npm install --silent zeromq 
RUN curl -L "https://github.com/docker/compose/releases/download/1.23.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose; chmod +x /usr/local/bin/docker-compose
COPY faas.js /server/faas.js
COPY build /server/build
EXPOSE 2000
EXPOSE 3333
CMD ["node", "/server/faas.js"]