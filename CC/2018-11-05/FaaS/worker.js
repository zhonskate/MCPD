var zmq = require('zeromq');

var sock = zmq.socket('pull');

sock.connect('tcp://127.0.0.1:2000');
console.log('Consumer connected to port 2000');

sock.on('message', function(msg){
    console.log('work: %s', msg.toString());
  });