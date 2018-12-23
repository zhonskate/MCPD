Estudiar la paralelizaci´on de Cholesky escalar para un tama˜no que, en secuencial tarde unos
60 segundos. Estudiar la evoluci´on del tiempo con respecto al n´umero de hilos utilizados (1,
2, 4, 8, 16 o 32).  
------> act3escalar.png

Realizar un estudio del tama˜no de bloque en el algoritmo de Cholesky por bloques. Tomad tres
valores de tama˜no de problema, variad el tama˜no de bloque en un rango lo suficientemente
amplio como para encontrar un tama˜no de bloque en el cual el tiempo sea m´ınimo para los
tres tama˜nos de problema. Todo ello en secuencial.  
------> resultstestblock.txt ------> se elige 250

Realizar una comparaci´on entre Cholesky escalar y por bloques en secuencial para diferentes
tama˜nos de problema.
------> act3compare.png
------> act3compare-log.png (eje y logarítmico)

Con un tama˜no de bloque fijo (el mejor) realizar un estudio experimental de Cholesky variando
el tama˜no de problema y el n´umero de hilos. Obtened tiempo de ejecuci´on, speedup y eficiencia.
------> act3final-new.png (tiempo)
------> act3final-su.png (speedup)
------> act3final-eff.png (eficiencia)

