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

var scaling_trigger = true;

const DB_NAME = 'db.json';
const COLLECTION_NAME = 'functions';
const UPLOAD_PATH = 'uploads';
const upload = multer({ dest: `${UPLOAD_PATH}/` }); // multer configuration
const db = new Loki(`${UPLOAD_PATH}/${DB_NAME}`, { persistenceMethod: 'fs' });

var jobq = new List();
var execlist = new List();
var workersq = new queue();

var app = express();
app.use(cors());
app.use(bodyParser.json());
var requestnum = 0;
var worker_replica_num = 1;
var worker_replicas = 1;
var registryIP = 'localhost';
var registryPort = '5000';
const address = process.env.ZMQ_BIND_ADDRESS || `tcp://*:2000`;
var functions = [];

var sock = zmq.socket('rep');
sock.bindSync(address);

console.log(`Broker serving on ${address}`);

//utils 

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

// REGISTER FUNCTIO

app.post('/registerfunction',upload.single('module'), async(req, res, next) =>{

    // receive from http
    try {
        // console.log(req);
        // add to database /uploads
        const col = await loadCollection(COLLECTION_NAME, db);
        const data = col.insert(req.file);
        db.saveDatabase();

        // create the sha of the tgz
        var tarfile = fs.readFileSync(req.file.path, 'utf8');
        var hash = sha256(tarfile);
        //console.log(hash);

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
            //if (stdout){console.log('stdout: ', stdout);}
            if (stderr){console.log('stderr: ', stderr);}
            // res.send(stdout);
            if (error !== null) {
                console.log('exec error: ', error);
                res.send(error);
                return next(new Error([error]));
            }

            var commandline = `\
            docker \
            tag \
            a/${hash} ${registryIP}:${registryPort}/a/${hash}`
            exec(commandline, function(error, stdout, stderr) {
                //if (stdout){console.log('stdout: ', stdout);}
                if (stderr){console.log('stderr: ', stderr);}
                // res.send(stdout);
                if (error !== null) {
                    console.log('exec error: ', error);
                    res.send(error); 
                    return next(new Error([error]));
                }

                var commandline = `\
                docker \
                push \
                ${registryIP}:${registryPort}/a/${hash}`
                exec(commandline, function(error, stdout, stderr) {
                    //if (stdout){console.log('stdout: ', stdout);}
                    if (stderr){console.log('stderr: ', stderr);}
                    // res.send(stdout);
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

// GET RESULT

app.get('/result/:reqnum', function (req, res) {
    if(req.params.reqnum <= requestnum){
        var df_path = `${__dirname}/requests/${req.params.reqnum}`;
        fs.readFile(`${df_path}/results.json`, function read(err, data) {
            if (err) {
                if(execlist.has(req.params.reqnum + "")){
                    res.send("REQUEST EXECUTING");
                }
                else{
                    res.send("REQUEST IN QUEUE");
                }
                //res.send(err);
            }
            res.send(data);
        });
    }
    else{
        res.send("NON-EXISTING REQUEST");
    }
});

app.get('/functionList', function (req, res) {
    res.send(functions);
});


// INVOKE FUNCTION

app.put('/invokefunction/:functionSha', function (req, res) {
    var timeStartReq = performance.now();
    requestnum = requestnum + 1;
    //console.log(req.body);
    // create the folder where the requests will be saved
    var df_path = `${__dirname}/requests/${requestnum}`;
    try{
        fs.mkdirSync(df_path);
    }
    catch(e){
        //console.log("directory already present")
    }

    // Initialize the data folders
    fs.writeFile(`${df_path}/params.json`, JSON.stringify(req.body), function(err) {
        if (err) {
            console.log(err);
        }
//        fs.writeFile(`${df_path}/results.json`, '', function(err) {
//            if (err) {
//                console.log(err);
//            }
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
        res.send({"requestnum":requestnum});
//        });
    });
});

app.listen(3333, function () {
  console.log('FaaS listening on port 3333!');
});

sock.on("message",function(msg){
    //dequeue from job queue
    stMsg = msg.toString();
    arrMsg = stMsg.split('///');
    if(arrMsg.length > 1){
        requestnumm = arrMsg[1];
        content = arrMsg[2];
        // console.log("CONTENT "+content);
        var df_path = `${__dirname}/requests/${requestnumm}`;
        fs.writeFile(`${df_path}/results.json`, content, function(err) {
            if (err) {
                console.log(err);
            }
            //console.log("RNNNNNNNNNN " + requestnumm);
            suc = execlist.delete(requestnumm + "");
            // console.log ("SUCC 1 " + suc);
            console.log("BUSY = " + arrMsg[4]);
            console.log("EXEC = " + arrMsg[5]);
            var timeEndReq = performance.now();
            var timeReq = timeEndReq - arrMsg[6];
            console.log("REQ = " + timeReq); 
        });

    }

    if(jobq.any()){
        job = jobq.shift();
        reqn = job.split("//")[0];      
        sendJob(reqn,job,msg);
    }
    else {workersq.enqueue(msg);}

});

function sendJob (rn,job,msg){
    console.log("DOING " + job);
    //add to doing queue
    execlist.push(rn + "");  // msg + '//' + job);
    // console.log("HEY  " + execlist.get(job));
    setTimeout(function(){ 
        //console.log("EXECLIST  " + rn);
        if (execlist.has(rn + "")){
            console.log("TOO LONG");
            execlist.delete(rn + "");
            //console.log ("SUCC 2 " + suc);
            if (workersq.isEmpty()){
                jobq.unshift(job);
            }
            else {
                msg=workersq.dequeue();
                sendJob(rn,job,msg);
            }
        }
    }, 10000);
    //console.log(execlist);
    //set timeout
    //send work
    sock.send(job);
}

function scaleUp(){
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
        workerfaas`;

    //    var commandline = `\
    //    docker run \ 
    //    -e "ZMQ_CONN_ADDRESS=tcp://broker:2000" \
    //    -v "/var/run/docker.sock:/var/run/docker.sock" \
    //    -v "/tmp/requests:/worker/requestsworker" \
    //    workerfaas`;
    
    var exec = require('child_process').exec;
    exec(commandline, function(error, stdout, stderr) {
        //if (stdout){console.log('stdout: ', stdout);}
        if (stderr){console.log('stderr: ', stderr);}
        // res.send(stdout);
        if (error !== null) {
            console.log('exec error: ', error);
        }
    });
    setTimeout(function(){ 
        if (jobq.length >=5 ){
            scaleUp();
        }
        else {
            scaling_trigger = true;
        }
    }, 30000);
}

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

process.on('SIGTERM', function() {
    console.log('Do something useful here.');
    for(cont_num =2; cont_num<=worker_replica_num; cont_num ++){
        console.log("killing container faas_worker_" + cont_num);
        var commandline = `\
        docker \
        kill \
        faas_worker_${cont_num} `;
        var exec = require('child_process').exec;
        exec(commandline, function(error, stdout, stderr) {
            //if (stdout){console.log('stdout: ', stdout);}
            if (stderr){console.log('stderr: ', stderr);}
            // res.send(stdout);
            if (error !== null) {
                console.log('exec error: ', error);
            }
            if (cont_num == worker_replica_num){
                process.exit();
            }
        });
    }
});
