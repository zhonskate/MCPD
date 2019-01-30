ZMQ:

el servidor pregunta a los clientes haciendo uso del patrón REQ/REP. Ellos responden, y ,una vez calculado el resulatado (la hora nueva) manda el nuevo valor haciendo uso del patrón PUB/SUB

Diagrama:

He hecho uso del reloj para mostrar el timeout. El comienzo de las interacciones es arbitrario, ya que se trata de un bucle. Sin embargo se mantiene el orden de las acciones dentro del bucle.
