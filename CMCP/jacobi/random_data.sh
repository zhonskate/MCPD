#!/bin/sh

echo "$1 $1"> matrix-data-jacobi.inp
echo "$1"> vector-data-jacobi.inp

INDEX=0
while [ $INDEX -lt $1 ]
do
    FOO=''
    INDEXJ=0
    while [ $INDEXJ -lt $1 ]
    do
        FOO="$FOO$(( ( RANDOM % 10 )  + 1 )) "
        INDEXJ=$(( $INDEXJ + 1 ))
    done
    echo "$FOO">> matrix-data-jacobi.inp
    INDEX=$(($INDEX + 1))
done

INDEX=0
while [ $INDEX -lt $1 ]
do
    FOO=''
    FOO="$FOO$(( ( RANDOM % 10 )  + 1 )) "
    echo "$FOO">> vector-data-jacobi.inp
    INDEX=$(($INDEX + 1))
done
