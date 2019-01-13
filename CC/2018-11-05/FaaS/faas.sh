#/bin/bash

if [ "$1" = "help" -o "$1" = "register" -o "$1" = "invoke" -o "$1" = "results" -o "$1" = "functions" ]
then
    if [ "$1" = "help" ]
    then
        echo "Usage ./faas <subcommand> (<options>)"
        echo ""
        echo "Subcommands:"
        echo "functions                              show functionsha list"
        echo "invoke <functionsha> <JSON> (<times>)  call a function, passing the json as input"
        echo "register <module.tar.gz>               register a function"
        echo "results <requestnum> (<end>)           display results of requests between requestnum and end"
        echo ""
        echo "for this tool to work the broker must be listening on the port 3333"
    fi
    if [ "$1" = "register" ]
    then
        if [ "$2" = "" ]
        then
            echo "USAGE register <module.tar.gz>"
        else
            ./scripts/register.sh $2
        fi
    fi
    if [ "$1" = "functions" ]
    then
        ./scripts/functions.sh
    fi
    if [ "$1" = "invoke" ]
    then
        if [ "$2" = "" ]
        then
            echo "USAGE invoke <functionsha> <JSON> (<times>)"
        else
            if [ "$3" = "" ]
            then
                echo "USAGE invoke <functionsha> <JSON> (<times>)"
            else
                if [ "$4" = "" ]
                then
                    ./scripts/invoke.sh $2 $3 1
                else                    
                    ./scripts/invoke.sh $2 $3 $4
                fi
            fi
        fi
    fi
    if [ "$1" = "results" ]
    then
        if [ "$2" = "" ]
        then
            echo "USAGE register <requestnum> (<end>)"
        else
            if [ "$3" = "" ]
            then
                ./scripts/result.sh $2 $2
            else
                ./scripts/result.sh $2 $3
            fi
        fi
    fi
else
    echo unrecognized subcommand. Use [./faas help] to show valid subcommands.
fi