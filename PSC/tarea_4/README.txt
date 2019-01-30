El frontend manda la información a los logs mediante un patrón pub/sub. 
El backend puede consultar información a los logs mediante el patrón req/rep, pero tambien puede obtener la información en forma de stream conectándose directamente al front mediante pub/sub.
