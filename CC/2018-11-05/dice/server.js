var fs = require("fs");

var mydata = JSON.parse(fs.readFileSync('/data/params.json', 'utf8'));

dices = 1|mydata.num_dices;
size = 6|mydata.size;
rollarr = [];
total = 0;

for (i=0;i<dices;i++){
    roll = Math.floor((Math.random() * size) + 1);
    total += roll;
    rollarr.push(roll);
}

var result = {
    "total": total,
    "results": rollarr
};

fs.writeFile("/data/results.json", JSON.stringify(result), (err) => {
    if (err) {
        console.error(err);
        return;
    };
});
