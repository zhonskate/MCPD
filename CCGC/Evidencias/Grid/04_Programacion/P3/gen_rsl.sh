#!/bin/bash
GASS_PORT=33209
LOCAL_HOST=javier.ccgc.mastercpd.upv.es
NUMBER_W=$1
# Creacion de los RSL
i=0
while [ $i -lt $NUMBER_W ]
do
echo "&" >script_$i.rsl
echo "(count = 1)" >>script_$i.rsl
echo "(executable = /bin/sh)" >>script_$i.rsl
echo "(arguments = compila_y_ejecuta_${i}.sh)" >>script_$i.rsl
echo "(rsl_substitution = (GASS_URL https://$LOCAL_HOST:${GASS_PORT}/home/ccgc/Evidencias/Grid/04_Programacion/P3))">>script_$i.rsl
echo "(stdout = \$(GASS_URL)/miStdout_${i})" >>script_$i.rsl
# echo "(stderr = \$(GASS_URL)/miStderr_${i})" >>script_$i.rsl
echo "(file_stage_in = ( \$(GASS_URL)/pi.c pi.c ) (\$(GASS_URL)/compila_y_ejecuta_${i}.sh compila_y_ejecuta_${i}.sh))">> script_$i.rsl

echo "#!/bin/bash" >compila_y_ejecuta_$i.sh
echo "gcc -o pi.ex pi.c" >> compila_y_ejecuta_$i.sh
echo "chmod +x pi.ex" >>compila_y_ejecuta_$i.sh
echo "./pi.ex ${i} 10 10000000000" >>compila_y_ejecuta_$i.sh
i=$(( i+1 ))
done
