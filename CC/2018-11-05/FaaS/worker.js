var zmq = require('zeromq');
var fs = require('fs');
var registryIP = 'localhost';
var registryPort = '5000';

var sock = zmq.socket('req');

sock.connect('tcp://127.0.0.1:2000');
console.log('Worker connected to port 2000');

sock.on('message', function(msg){
    console.log("WORKER1 EXECUTING");
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
        fs.writeFile(`${df_path}/results.json`, '', function(err) {
            if (err) {
                console.log(err);
            }
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
                    //console.log('stdout: ', stdout);
                }
                if(stderr){
                    console.log('stderr: ', stderr);
                }
                if (error !== null) {
                    console.log('exec error: ', error);
                }
                fs.readFile(`${df_path}/results.json`, function read(err, data) {
                    if (err) {
                        throw err;
                    }
                    content = data;
                
                    // Invoke the next step here however you like
                    console.log(content);   // Put all of the code here (not the best solution)                       
                    console.log("WORKER1 WAITING");
                    sock.send("worker1" + '///' + requestnum + '///' + content + '///' + stMsg);
                });
            });
        });
    });
  });
  console.log("WORKER1 WAITING");
  sock.send("worker1");