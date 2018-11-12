var express = require('express');
var fs = require('fs');
var bodyParser = require('body-parser');
var multer = require('multer');
var cors = require('cors');
var path = require('path');
var Loki = require('lokijs');
var del = require('del');
var sha256 = require('js-sha256');

const DB_NAME = 'db.json';
const COLLECTION_NAME = 'functions';
const UPLOAD_PATH = 'uploads';
const upload = multer({ dest: `${UPLOAD_PATH}/` }); // multer configuration
const db = new Loki(`${UPLOAD_PATH}/${DB_NAME}`, { persistenceMethod: 'fs' });

var app = express();
app.use(cors());
app.use(bodyParser.json());
var requestnum = 0;

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


// REGISTER FUNCTION

app.post('/registerfunction',upload.single('module'), async(req, res) =>{

    // receive from http
    try {

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
            }
        });

        // return the sha
        res.send(hash);
    } catch (err) {
        res.sendStatus(400);
    }
});


// INVOKE FUNCTION

app.post('/invokefunction/:functionSha', function (req, res) {
    console.log(req.body);

    // create the folder where the requests will be saved
    var df_path = `${__dirname}/requests/${requestnum}`;
    try{
        fs.mkdirSync(df_path);
    }
    catch(e){
        console.log("directory already present")
    }

    // Initialize the data folders
    fs.writeFile(`${df_path}/params.json`, JSON.stringify(req.body), function(err) {
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
    a/${req.params.functionSha} \
    npm start`;

    var exec = require('child_process').exec;
    exec(commandline, function(error, stdout, stderr) {
        if(stdout){
            console.log('stdout: ', stdout);
        }
        if(stderr){
            console.log('stderr: ', stderr);
        }
        res.send(stdout);
        if (error !== null) {
            console.log('exec error: ', error);
        }
    });
    requestnum = requestnum + 1;

});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});