#!/bin/bash
GASS_PORT=38648
LOCAL_HOST=javier.ccgc.mastercpd.upv.es
NUMBER_IMAGES=$1
# Creacion de los RSL
i=0
while [ $i -lt $NUMBER_IMAGES ]
do
echo "&" >script_$i.rsl
echo "(count = 1)" >>script_$i.rsl
echo "(executable = /usr/bin/convert)" >>script_$i.rsl
echo "(arguments = -paint 5 img0$i.jpg img0${i}_f.jpg )" >>script_$i.rsl
echo "(rsl_substitution = (GASS_URL https://$LOCAL_HOST:${GASS_PORT}/home/ccgc/Evidencias/Grid/04_Programacion/P1))" >>script_$i.rsl
echo "(stdout = \$(GASS_URL)/miStdout)" >>script_$i.rsl
echo "(stderr = \$(GASS_URL)/miStderr)" >>script_$i.rsl
echo "(file_stage_in=( \$(GASS_URL)/input_images/img0$i.jpg img0$i.jpg ) )" >>script_$i.rsl
echo "(file_stage_out=(img0${i}_f.jpg \$(GASS_URL)/output_images/img0${i}_f.jpg) )" >>script_$i.rsl
i=$(( i+1 ))
done
