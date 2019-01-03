var zmq = require('zeromq');
var fs = require('fs');
var registryIP = 'localhost';
var registryPort = '5000';

var sock = zmq.socket('pull');

sock.connect('tcp://127.0.0.1:2000');
console.log('Worker connected to port 2000');

sock.on('message', function(msg){
    stMsg = msg.toString();
    console.log('work: %s', stMsg);
    arrMsg = stMsg.split('//');
    requestnum = arrMsg[0];
    functionSha = arrMsg[1];
    reqJson = arrMsg[2];
    console.log(reqJson);

    // create the folder where the requests will be saved
    var df_path = `${__dirname}/requestsworker/${requestnum}`;
    try{
        fs.mkdirSync(df_path);
    }
    catch(e){
        console.log("directory already present")
    }

    // Initialize the data folders
    fs.writeFile(`${df_path}/params.json`, reqJson, function(err) {
        if (err) {
            console.log(err);
        }
    });

    fs.writeFile(`${df_path}/results.json`, '', function(err) {
        if (err) {
            console.log(err);
        }
    });

    // Run the container
    var commandline = `\
    docker \
    run \
    --rm \
    -w /workdir/sum \
    -v ${df_path}/params.json:/data/params.json \
    -v ${df_path}/results.json:/data/results.json \
    ${registryIP}:${registryPort}/a/${functionSha} \
    npm start`;

    var exec = require('child_process').exec;
    exec(commandline, function(error, stdout, stderr) {
        if(stdout){
            console.log('stdout: ', stdout);
        }
        if(stderr){
            console.log('stderr: ', stderr);
        }
        if (error !== null) {
            console.log('exec error: ', error);
        }
        
    });

  });