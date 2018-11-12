var fs = require("fs");

var mydata = JSON.parse(fs.readFileSync('/data/params.json', 'utf8'));

var result = {
    result: mydata.a + mydata.b
};

fs.writeFile("/data/results.json", JSON.stringify(result), (err) => {
    if (err) {
        console.error(err);
        return;
    };
});
