

function porDos(n,callback){

    console.log("soy porDos()\n");
    callback( n*2 );
}

function main(){

    console.log("soy main()\n");
    porDos(7,function(res){
        console.log("7 por dos da "+ res);
    }
    );

}

main();