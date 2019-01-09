FROM node
COPY worker.js /worker/worker.js
WORKDIR /worker
RUN npm install --silent zeromq
EXPOSE 2000
CMD ["node", "/worker/worker.js"]