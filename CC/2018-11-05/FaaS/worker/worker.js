var zmq = require('zeromq');
var fs = require('fs');
var registryIP = 'localhost';
var registryPort = '5000';
const { PerformanceObserver, performance } = require('perf_hooks');

var sock = zmq.socket('req');
const address = process.env.ZMQ_CONN_ADDRESS || `tcp://127.0.0.1:2000`;

sock.connect(address);
console.log(`Worker connected to ${address}`);

sock.on('message', function(msg){
    var timeStartBusy = performance.now();
    console.log("EXECUTING");
    stMsg = msg.toString();
    console.log('work: %s', stMsg);
    arrMsg = stMsg.split('//');
    requestnum = arrMsg[0];
    functionSha = arrMsg[1];
    reqJson = arrMsg[2];
    // console.log(reqJson);

    // create the folder where the requests will be saved
    var df_path = `/worker/requestsworker/${requestnum}`;
    var local_path = `/tmp/requests/${requestnum}`;
    try{
        fs.mkdirSync(df_path);
    }
    catch(e){
        console.log("directory already present")
    }

    // Initialize the data folders
    fs.writeFile(`${df_path}/params.json`, reqJson, function(err) {
        if (err) {
            console.log(err.stack);
        }
        fs.writeFile(`${df_path}/results.json`, '', function(err) {
            if (err) {
                console.log(err.stack);
            }
            var timeStartExec = performance.now();
            // Run the container
            var commandline = `\
            docker \
            run \
            --rm \
            -w /workdir \
            -v ${local_path}/params.json:/data/params.json \
            -v ${local_path}/results.json:/data/results.json \
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
                    // console.log(content);   // Put all of the code here (not the best solution)    
                    var timeEndExec = performance.now();                  
                    console.log("WAITING");
                    var timeExec = timeEndExec - timeStartExec;
                    var timeEndBusy = performance.now(); 
                    var timeBusy = timeEndBusy - timeStartBusy;
                    sock.send("worker1" + '///' + requestnum + '///' + content + '///' + stMsg + '///' + timeBusy + '///' + timeExec + '///' + arrMsg[3]);
                });
            });
        });
    });
  });
  console.log("WAITING");
  sock.send("worker1");