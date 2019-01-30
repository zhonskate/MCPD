Mi sistema se compone de un nodo servidor y nodos clientes que se conectan a él.
Se pueden realizar comunicaciones entre un par de clientes, todos ellos o un subgrupo.
El server bindea un socket pull y un socket pub. El pull recibe los mensajes y el pub los envía.
Todos los clientes estan suscritos a su propio identificador único, por lo que cualquier cliente puede mandar un mensaje unicamente a otro cliente.
Para mandar un mensaje a un subgrupo, se mandará previamente un mensaje unico a todos los clientes que pertenecerán al grupo indicándoles que se suscriban a ese identificador. Luego se mandará el mensaje.  
Este sistema tiene un punto único de fallo, que es el servidor.
