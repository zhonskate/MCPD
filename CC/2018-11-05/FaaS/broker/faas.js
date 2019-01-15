// dependencies

var express = require('express');
var List = require("collections/list");
var queue = require('queue-fifo');
var fs = require('fs');
var bodyParser = require('body-parser');
var multer = require('multer');
var cors = require('cors');
var path = require('path');
var Loki = require('lokijs');
var del = require('del');
var sha256 = require('js-sha256');
var zmq = require('zeromq');
const { PerformanceObserver, performance } = require('perf_hooks');

// Db related declarations

const DB_NAME = 'db.json';
const COLLECTION_NAME = 'functions';
const UPLOAD_PATH = 'uploads';
const upload = multer({ dest: `${UPLOAD_PATH}/` }); // multer configuration
const db = new Loki(`${UPLOAD_PATH}/${DB_NAME}`, { persistenceMethod: 'fs' });

// Data structures

var jobq = new List();
var execlist = new List();
var workersq = new queue();
var functions = [];

// Express configuration

var app = express();
app.use(cors());
app.use(bodyParser.json());

// other declarations

var scaling_trigger = true;
var requestnum = 0;
var worker_replica_num = 1;
var worker_replicas = 1;
var registryIP = 'localhost';
var registryPort = '5000';
const address = process.env.ZMQ_BIND_ADDRESS || `tcp://*:2000`;

// zmq init

var sock = zmq.socket('rep');
sock.bindSync(address);

console.log(`Broker serving on ${address}`);

//----------------------------------------------------------------------------------//
// Upload-related code

const loadCollection = function (colName, db){
    return new Promise(resolve => {
        db.loadDatabase({}, () => {
            const _collection = db.getCollection(colName) || db.addCollection(colName);
            resolve(_collection);
        })
    });
}

const cleanFolder = function (folderPath) {
    // delete files inside folder but not the folder itself
    del.sync([`${folderPath}/**`, `!${folderPath}`]);
};

cleanFolder(UPLOAD_PATH);

//----------------------------------------------------------------------------------//
// REGISTER FUNCTION

app.post('/registerfunction',upload.single('module'), async(req, res, next) =>{

    // receive from http
    try {
        // add to database /uploads
        const col = await loadCollection(COLLECTION_NAME, db);
        const data = col.insert(req.file);
        db.saveDatabase();

        // create the sha of the tgz
        var tarfile = fs.readFileSync(req.file.path, 'utf8');
        var hash = sha256(tarfile);

        // prepare folder to build the image
        fs.rename(req.file.path,'./build/module.tar.gz',function(error, stdout, stderr){
            if(error){console.log(error);}
            if(stderr){console.log(stderr);}
            if(stdout){console.log(stdout);}
        })

        // build the docker image 
        var df_path = "./build/";
        var commandline = `\
        docker \
        build \
        -t a/${hash} \
        ${df_path}`;
        var exec = require('child_process').exec;
        exec(commandline, function(error, stdout, stderr) {

            if (stderr){
                console.log('stderr: ', stderr);
            }

            if (error !== null) {
                console.log('exec error: ', error);
                res.send(error);
                return next(new Error([error]));
            }

            // tag the docker image to upload it
            var commandline = `\
            docker \
            tag \
            a/${hash} ${registryIP}:${registryPort}/a/${hash}`
            exec(commandline, function(error, stdout, stderr) {

                if (stderr){
                    console.log('stderr: ', stderr);
                }

                if (error !== null) {
                    console.log('exec error: ', error);
                    res.send(error); 
                    return next(new Error([error]));
                }
                
                // push the image to the registry
                var commandline = `\
                docker \
                push \
                ${registryIP}:${registryPort}/a/${hash}`
                exec(commandline, function(error, stdout, stderr) {

                    if (stderr){
                        console.log('stderr: ', stderr);
                    }

                    if (error !== null) {
                        console.log('exec error: ', error);
                        res.send(error); 
                        return next(new Error([error]));
                    }

                    // return the sha
                    functions.add(hash);
                    res.send(hash);

                });
            });
        });

    } catch (err) {
        console.log(err);
        res.sendStatus(400);
    }
});

//----------------------------------------------------------------------------------//
// GET RESULT

app.get('/result/:reqnum', function (req, res) {
    
    if(req.params.reqnum <= requestnum){

        var df_path = `${__dirname}/requests/${req.params.reqnum}`;

        // read result file 
        fs.readFile(`${df_path}/results.json`, function read(err, data) {
            if (err) {
                if(execlist.has(req.params.reqnum + "")){
                    res.send("REQUEST EXECUTING");
                }
                else{
                    res.send("REQUEST IN QUEUE");
                }
            }
            res.send(data);
        });
    }
    else{
        res.send("NON-EXISTING REQUEST");
    }
});

//----------------------------------------------------------------------------------//
// GET FUNCTION LIST

app.get('/functionList', function (req, res) {
    res.send(functions);
});

//----------------------------------------------------------------------------------//
// INVOKE FUNCTION

app.put('/invokefunction/:functionSha', function (req, res) {

    // TODO fix this time
    var timeStartReq = performance.now();
    requestnum = requestnum + 1;

    // create the folder where the requests will be saved
    var df_path = `${__dirname}/requests/${requestnum}`;
    try{
        fs.mkdirSync(df_path);
    }
    catch(e){

    }

    // Initialize the data folders
    fs.writeFile(`${df_path}/params.json`, JSON.stringify(req.body), function(err) {
        if (err) {
            console.log(err);
        }

        // enqueue or assign the job
        console.log('enqueuing request nº ' + requestnum);
        job = requestnum + '//'+ req.params.functionSha + '//' + JSON.stringify(req.body) + '//' + timeStartReq ;
        if (workersq.isEmpty()){
            jobq.push(job);
            if(scaling_trigger && jobq.length >=5){
                scaling_trigger = false;
                scaleUp();
            }
        }
        else {
            msg=workersq.dequeue();
            sendJob(requestnum,job,msg);
        }
        // return request num
        res.send({"requestnum":requestnum});
    });
});

//----------------------------------------------------------------------------------//
// START EXPRESS

app.listen(3333, function () {
  console.log('FaaS listening on port 3333!');
});

//----------------------------------------------------------------------------------//
// MESSAGE HANDLER

sock.on("message",function(msg){

    //read message
    stMsg = msg.toString();
    arrMsg = stMsg.split('///');

    // returns from working
    if(arrMsg.length > 1){
        requestnumm = arrMsg[1];
        content = arrMsg[2];
        var df_path = `${__dirname}/requests/${requestnumm}`;

        // write the results file
        fs.writeFile(`${df_path}/results.json`, content, function(err) {
            if (err) {
                console.log(err);
            }
            suc = execlist.delete(requestnumm + "");
            
            //console.log("BUSY = " + arrMsg[4]);
            //console.log("EXEC = " + arrMsg[5]);
            var timeReq = performance.now() - arrMsg[6];
            //console.log("REQ = " + timeReq); 
            console.log(requestnumm + "\t" + worker_replicas + "\t" + timeReq + "\t" + arrMsg[4]);
        });

    }

    // send work or enqueue worker
    if(jobq.any()){
        job = jobq.shift();
        reqn = job.split("//")[0];      
        sendJob(reqn,job,msg);
    }
    else {workersq.enqueue(msg);}

});

//----------------------------------------------------------------------------------//
// SEND JOB

function sendJob (rn,job,msg){

    console.log("DOING REQUEST " + rn);

    //add to doing queue
    execlist.push(rn + "");  

    // retry if it gets stuck
    setTimeout(function(){ 
        if (execlist.has(rn + "")){
            console.log("REQUEST " + rn + " TOOK TOO LONG, RETRYING");
            execlist.delete(rn + "");
            if (workersq.isEmpty()){
                jobq.unshift(job);
            }
            else {
                msg=workersq.dequeue();
                sendJob(rn,job,msg);
            }
        }
    }, 60000);

    sock.send(job);
}

//----------------------------------------------------------------------------------//
// SCALE UP

function scaleUp(){

    // update variables 
    worker_replica_num = worker_replica_num + 1;
    worker_replicas = worker_replicas + 1;
    console.log ("SCALING UP TO " + worker_replicas + " REPLICAS");

    //compose_file = "/Users/zhon/Documents/MCPD/CC/2018-11-05/FaaS/docker-compose.yml"
    //var commandline = `\
    //    docker-compose -f ${compose_file} up \
    //    --scale worker=${worker_replicas} \
    //    -d `;

    var commandline = `\
        docker \
        run \
        -d \
        --name faas_worker_${worker_replica_num} \
        --rm \
        --network faas_default \
        -e ZMQ_CONN_ADDRESS=tcp://broker:2000 \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v /tmp/requests:/worker/requestsworker \
        jrodriguez96/workerfaas`;
    
    // run the worker container
    var exec = require('child_process').exec;
    exec(commandline, function(error, stdout, stderr) {
        if (stderr){console.log('stderr: ', stderr);
    }
        if (error !== null) {
            console.log('exec error: ', error);
        }
    });

    // control the scaling rate
    setTimeout(function(){ 
        if (jobq.length >=5 ){
            scaleUp();
        }
        else {
            scaling_trigger = true;
        }
    }, 30000);
}

//----------------------------------------------------------------------------------//
// SCALE DOWN

// scale down policies
var scaleinRetries = 0;
setInterval(function(){
    if(!workersq.isEmpty()){
        scaleinRetries = scaleinRetries + 1;
    }
    else{
        scaleinRetries = 0;
    }
    if (scaleinRetries >= 5){
        scaleinRetries = 0;
        scaleDown();
    }
},5000);

function scaleDown(){

    if(worker_replicas>1){
        worker_replicas=worker_replicas-1;
        console.log ("SCALING DOWN TO " + worker_replicas + " REPLICAS");
        sock.send("KILL");
    }
    else{
        console.log("CANNOT SCALE DOWN, ONLY 1 REPLICA");
    }
}

//----------------------------------------------------------------------------------//
// SIGNAL HANDLING

process.on('SIGTERM', function() {
    console.log('Received SIGTERM');

    // kill all the workers
    for(cont_num =2; cont_num<=worker_replica_num; cont_num ++){
        console.log("killing container faas_worker_" + cont_num);
        var commandline = `\
        docker \
        kill \
        faas_worker_${cont_num} `;
        var exec = require('child_process').exec;
        exec(commandline, function(error, stdout, stderr) {
            if (stderr){console.log('stderr: ', stderr);}
            if (error !== null) {
                console.log('exec error: ', error);
            }
            if (cont_num == worker_replica_num){
                process.exit();
            }
        });
    }
});
