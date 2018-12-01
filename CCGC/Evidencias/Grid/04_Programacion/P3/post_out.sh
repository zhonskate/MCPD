cat miStdout_* > miStdoutcombined
awk '{s+=$1}END{printf "%.20f\n", s}' miStdoutcombined
rm miStdoutcombined
