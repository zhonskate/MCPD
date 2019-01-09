FROM node
COPY worker.js /worker/worker.js
WORKDIR /worker
RUN mkdir requestsworker
RUN npm install --silent zeromq
EXPOSE 2000
CMD ["node", "/worker/worker.js"]