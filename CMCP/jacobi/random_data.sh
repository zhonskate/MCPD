#!/bin/sh

echo "$1 $1"> $1-matrix-data-jacobi.inp
echo "$1"> $1-vector-data-jacobi.inp

INDEX=0
while [ $INDEX -lt $1 ]
do
    FOO=''
    INDEXJ=0
    while [ $INDEXJ -lt $1 ]
    do
	if [ $INDEX -eq $INDEXJ ]
	then
	    FOO="$FOO $(( ( RANDOM % (( $1 * $1 )) )  + $1 )).$((  (( ( RANDOM % 10000 )  + 1 )) % 1000  )) "
	else
            FOO="$FOO 0.$((  (( ( RANDOM % 10000 )  + 1 )) % 1000  )) "
	fi
        INDEXJ=$(( $INDEXJ + 1 ))
    done
    echo "$FOO">> $1-matrix-data-jacobi.inp
    INDEX=$(($INDEX + 1))
done

INDEX=0
while [ $INDEX -lt $1 ]
do
    FOO=''
    FOO="$FOO$(( ( RANDOM % 10 )  + 1 )).$((  (( ( RANDOM % 10000 )  + 1 )) % 1000  )) "
    echo "$FOO">> $1-vector-data-jacobi.inp
    INDEX=$(($INDEX + 1))
done
