var express = require('express');
var app = express();
var fs = require('fs');
var bodyParser = require('body-parser');

app.use(bodyParser.json());

app.get('/registerfunction', function (req, res) {
    // receive from http
    // build
    var df_path = "/Users/zhon/Documents/MCPD/CC/2018-11-05/FaaS";
    var commandline = `\
    docker \
    build \
    -t prueba1 \
    ${df_path}`;

    var exec = require('child_process').exec;
    exec(commandline, function(error, stdout, stderr) {
        console.log('stdout: ', stdout);
        console.log('stderr: ', stderr);
        res.send(stdout);
        if (error !== null) {
            console.log('exec error: ', error);
            res.send(error);
        }
    });
});

app.post('/invokefunction', function (req, res) {
    console.log(req.body);
    var df_path = "/Users/zhon/Documents/MCPD/CC/2018-11-05/FaaS";
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
    var commandline = `\
    docker \
    run \
    --rm \
    -w /workdir/module \
    -v ${df_path}/params.json:/data/params.json \
    -v ${df_path}/results.json:/data/results.json \
    prueba1 \
    npm test`;

    var exec = require('child_process').exec;
    exec(commandline, function(error, stdout, stderr) {
        console.log('stdout: ', stdout);
        console.log('stderr: ', stderr);
        res.send(stdout);
        if (error !== null) {
            console.log('exec error: ', error);
        }
    });

});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});