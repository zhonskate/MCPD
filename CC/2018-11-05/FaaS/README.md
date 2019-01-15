# Diseño e implementación de un FaaS

## Índice

[1.- Introducción](#1-introducción)

[2.- Diseño](#2-diseño)

- [2.1.- Diseño del broker](#21-diseño-del-broker)
- [2.2.- Diseño del worker](#22-diseño-del-worker)

[3.- Implementación](#3-implementación)

- [3.1.- Tecnologías utilizadas](#31-tecnologías-utilizadas)
- [3.2.- Detalles de implementación](#22-detalles-de-implementación)

[4.- Despliegue](#4-despliegue)

[5.- Pruebas](#5-pruebas)

- [5.1.- Pruebas de validación](#51-pruebas-de-validación)
- [5.2.- Pruebas de rendimiento](#52-pruebas-de-rendimiento)

[6.- Trabajos Futuros](#6-trabajos-futuros)

[7.- Conclusiones](#7-conclusiones)

## 1.- Introducción

En el trabajo siguiente se ha realizado el diseño y desarrollo de un *FaaS* (Function as a service). Se trata de un modelo de plataforma que permite abstraer a los clientes que van a desarrollar, lanzar y gestionar aplicaciones de la infraestructura subyacente, de la cual se tendrían que preocupar en un modelo de desarrollo clásico. Cuando se crea una aplicación siguiendo este modelo, se consigue una arquitectura *serverless*, típicamente utilizada en aplicaciones basadas en microservicios. 

Existen multitud de productos *FaaS* disponibles para ser utilizados. Algunos ejemplos son [AWS Lambda](https://aws.amazon.com/es/lambda/features/), [Google Cloud Functions](https://cloud.google.com/functions/) o [OpenFaaS](https://docs.openfaas.com/).

A continuación se describirá la propuesta realizada, pasando por su diseño, implementación, modo de despliegue, algunas pruebas de validación y rendimiento y posibles mejoras y trabajos futuros.

## 2.- Diseño

![grand-scheme](/md-images/grand-scheme.png)
>Esquema general del sistema

El sistema se compone principalmente de tres piezas interconectadas. Estas piezas son el [registry](https://docs.docker.com/registry/), un almacén de imágenes docker; el *broker*, que administrará tanto el uso del sistema mediante una *API REST* como la repartición de tareas entre los trabajadores o *workers*; y, por último, los propios workers, que serán los encargados de realizar las ejecuciones en última instancia y de etregar el resultado de vuelta al broker. Tal y como se puede observar en la imagen anterior, la conexión entre el broker y los workers se realizará haciendo uso de [ZeroMQ](http://zeromq.org/). 

Como el registry utilizado se obtiene de la [imagen oficial de docker hub](https://hub.docker.com/_/registry), su diseño es ajeno a este trabajo. A continuación se mostrará el diseño de los otros dos componentes.

### 2.1.- Diseño del broker

El broker debe cumplir principalmente cuatro tareas: Gestionar las llamadas de los usuarios, es decir, la API; Encargarse de la construcción y subida de las imágenes al *registry*; Enviar el trabajo a los workers y gestionar el escalado de éstos.

* **Gestionar las llamadas de los usuarios:** El broker va a ser el encargado de recibir y devolver las peticiones de los usuarios, es decir, es la puerta de entrada al sistema. Para ser accesible desde el exterior se hace uso de una *API REST*. Esta API tiene cuatro funciones disponibles:

    * *Registrar función*: El usuario debe ser capaz de proveer el código de una función para poder hacer posteriormente un uso de ella.

    * *Invocar función*: Una vez la función ha sido subida, tiene que ser accesible. Esta función tiene que ser capaz de recibir parámetros.

    * *Obtener resultados*: Para mantener la asincronía del sistema, el broker no tiene que mandar el resultado cuando una ejecución finaliza. Es deber del usuario preocuparse por el estado de sus ejecuciones. Por tanto, el broker se encargará de devolver el estado de una ejecución cuando sea preguntado, mediante esta función.

    * *Obtener funciones*: En caso de que un usuario quiera obtener un listado de las funciones disponibles, esta función las devolverá.

* **Encargarse de la construcción y subida de imágenes:** Este punto está muy relacionado con la primera de las funciones de la API. Cuando se recibe el código y dependencias de una función, el broker tiene que ser capaz de construir una imagen *docker* y subirla al registro, con el objetivo de poder ser utilizada más tarde por todos los workers. 

* **Enviar el trabajo a los workers:** El broker debe ser capaz de enviar el trabajo, es decir, los identificadores de las funciones y los parámetros de entrada, a los workers para su ejcución. Para lograrlo se optará por un patrón en el que son los workers los que inician la comunicación, avisando al broker de que están disponibles para trabajar. En ese momento, el broker les enviará el trabajo y, una vez completado, los workers devolverán el resultado.  Al utilizar este patrón de comunicación surgen diversos problemas, como cómo almacenar los trabajos que no están siendo ejecutados, cómo identificar a los trabajadores ocupados y disponibles y cómo detectar el estado de un trabajo, relacionado con la tercera función de la API. Para solventar estos problemas, se han diseñado tres estructuras de datos que serán implementadas en el worker:

    * *Cola de trabajos*: Soluciona el primero de los problemas, pues los trabajos que no puedan ser asignados inmediatamente serán encolados, a la espera de que quede libre un worker que se pueda encargar de su ejecución.

    * *Pool de Workers*: Soluciona la forma de identificar a los workers disponibles, pues son aquellos que se encuentran en la *pool*. Cuando haya trabajo encolado, el worker lo cogerá inmediatamente. Cuando no haya trabajo en la cola, el worker se unirá a la *pool*.

    * *Lista de trabajos activos*: Cuando un trabajo pasa a ser ejecutado, es decir, se manda a un *worker*, se elimina de la *cola de trabajos* y se añade a esta lista. Una vez se completa se elimina de aquí. En el caso de que la ejecución no se complete en un tiempo establecido, saltaría un timeout y el trabajo se volvería a encolar. Consultando esta estructura y la cola se puede saber el estado del trabajo en cuestión.

* **Gestionar el escalado de los workers:**  El broker debe tener la capacidad de aumentar y disminuir el numero de workers en función de la carga recibida. De esta manera se puede lograr alcanzar un equilibrio entre el tiempo de respuesta de las peticiones y la carga de procesamiento del sistema. Es decir, minimizar el tiempo de respuesta utilizando la mínima capacidad de cómputo posible. 

### 2.2.- Diseño del worker

La función principal del worker es la ejecución de los trabajos obtenidos del broker. Para llevar a cabo esta ejecución tiene que ser capaz de obtener las imágenes en cuestión, es decir, tener acceso al registro y poder lanzar contenedores docker hermanos. Por otro lado, tiene que ser capaz de pasar los parámetros para la ejecución a dichos contenedores función y recoger sus resultados para devolverlos al broker.

## 3.- Implementación

El progreso de la implementación (Los *commits* antiguos) se pueden ver [aquí](https://github.com/zhonskate/MCPD/tree/master/CC/2018-11-05/FaaS).

### 3.1.- Tecnologías utilizadas

El proyecto se ha desarrollado principalmente en [nodejs](https://nodejs.org/es/). El código está compuesto de archivos *JavaScript*, *Dockerfiles* para la creación de imágenes y archivos *YAML* para ser utilizados por docker-compose. En cuanto a la comunicación broker-worker se hace uso de *ZeroMQ*. A continuación se presenta una lista más exhaustiva de tecnologías.

* [nodejs](https://nodejs.org/es/) Como plataforma de desarrollo.

* [express](https://www.npmjs.com/package/express) Para la implementación de la API REST.

* [queue-fifo](https://www.npmjs.com/package/queue-fifo) Para la implementación del *pool* de workers.

* [multer](https://www.npmjs.com/package/multer),[body-parser](https://www.npmjs.com/package/body-parser), [cors](https://www.npmjs.com/package/cors) y [lokijs](https://www.npmjs.com/package/lokijs) Para la subida de archivos.

* [js-sha256](https://www.npmjs.com/package/js-sha256) Para la creación de identificadores

* [zeromq](http://zeromq.org/) Para la comunicación entre el broker y los workers.

* [docker](https://www.docker.com/) Para la gestión de componentes y la invocación de contenedores.

* [docker-compose](https://docs.docker.com/compose/) Para la orquestación y el despliegue de los componentes.

### 3.2.- Detalles de implementación

A la hora de revisar los detalles de la implementación se va a seguir una estructura similar a la seguida en la sección [2](#2-diseño).

#### 3.2.1.- Detalles de la implementación del broker

* **Gestionar las llamadas de los usuarios:** Como se comentaba en la sección [2.1](#21-diseño-del-broker), las llamadas de los usuarios se van a gestionar mediante una *API REST*. Esta API ha sido implementada haciendo uso de la herramienta *express* y ofrece cuatro funciones:

    * `/registerfunction`. Se trata de un método POST que recibe un *.tar.gz*, lo convierte en una imagen docker y lo sube al registry. Este *.tar.gz* debe ser un módulo *npm* comprimido, con un *package.json* con las dependencias necesarias y cuyo archivo principal se debe llamar *server.js*, pues a la hora de ejecutar la función, el worker ejecutará `npm start` en el contenedor función final. Este método le devuelve al usuario un *SHA256*, que deberá utilizar más adelante para llamar a la función registrada.

    * `/invokefunction/:functionSha`. Se trata de un método PUT que recibe como parámetro un *JSON*, en el que se encontrarán los valores de entrada de la función a invocar. Este método devuelve un número de petición o *requestnum*, que será el utilizado por el usuario para comprobar el estado de esta ejecución.

    * `/result/:reqnum`. Se trata de un método GET donde se indica en la *URL* el estado de qué ejecución se quiere comprobar. El sistema devolverá el resultado, en el caso de que la ejecución haya terminado; un mensaje *Ejecutando*, en el caso de que ya haya sido asignado a un worker pero todavía no haya acabado la ejecución; un mensaje *En cola* en el caso de que ya haya sido aceptada la petición pero todavía no haya sido enviada a un worker para su ejecución o un mensaje *Petición no disponible* en el caso de que no exista tal petición.

    * `/functionList`. Se trata de un método GET que devuelve una *array* con los *SHA256* de las funciones registradas en el sistema.

* **Encargarse de la construcción y subida de imágenes:** Tal y como se comenta en el primer punto de la lista anterior, se crea una imagen docker haciendo uso de un template (disponible en `/broker/build/Dockerfile`) y se sube al registry.

![worker](/md-images/zmq.png)
> Patrón de interacción *zeromq* entre el broker y los workers

* **Enviar el trabajo a los workers:** Como se ha indicado en el apartado [2.1](#21-diseño-del-broker), se crean tres estructuras de datos para manejar el estado de los workers y los trabajos. Son las siguientes:

    * *Cola de trabajos*: La cola de trabajos se implementa como una lista, ya que, en el caso de que un trabajo no termine de ejecutarse en el tiempo aceptable (y salte su timeout), tiene que volver a incorporarse a la lista, y, es conveniente que lo haga en la primera posición, pues así se minimiza el tiempo total de esa petición que ha fallado.

    * *Pool de Workers*: Se implementa como una cola, aunque, como los workers no están identificados, no se tiene porqué enviar el trabajo siguiente al primer worker encolado. El envío se gestiona automáticamente mediante el socket *reply* de *zeromq*.

    * *Lista de trabajos activos*: Se implementa como una lista, pues los elementos se añaden y eliminan en base a su identificador.

    Un trabajo se puede enviar a un worker al ocurrir cualquiera de estos dos eventos:

    * **Llega un mensaje de un worker** En el caso de que la cola de trabajos no esté vacía, el trabajo será asignado directamente al worker, pues los workers solo envían mensajes al broker cuando no tienen trabajo, bien sea porque se acaban de encender o porque acaban de terminar la ejecución anterior. En el caso de que la cola de trabajos esté vacía, el worker se añadiría a la *pool* de workers. Como se implementa un patrón request-reply entre el los workers y el broker, el mensaje request se quedaría pendiente a la espera de que se generara un reply por parte del broker. Es por esto que la cola de workers no es necesaria, pues se eligen en base a las asignaciones que realiza el patrón *zeromq*.

    * **Llega una llamada a invokefunction** De forma análoga al caso anterior, cuando llega una petición para ejecutar una función, se pueden dar estas dos situaciones: Si la cola pool de workers no está vacía, se asigna el trabajo a un worker disponible. En el caso de estar vacía, el trabajo se encola a la espera de que un worker quede libre.

![worker](/md-images/broker-exec.png)
> Esquema del envío de un trabajo 

* **Gestionar el escalado de los workers:** Para gestionar el escalado de los workers se han definido dos funciones. Una función `scaleUp()` que añade un worker y una función `scaleDown()`, que elimina un worker. Estas funciones llaman desde dentro del contenedor en el que se encuentra el broker a el demonio docker del host, pues esto se consigue realizando un *binding* del socket de docker del host al contenedor. Este escalado se podría realizar compartiendo los binarios de docker-compose entre el host y el contenedor, pero como este proyecto ha sido desarrollado en macOS, *docker for mac* no lo permite. De esta manera, al levantar una instancia nueva de la imagen del worker, ésta tiene que añadirse a la red creada por el despliegue inicial, de forma que el broker la pueda encontrar. También se debe seguir cierta nomenclatura al declarar las imágenes, es por eso que más adelante se pedirá, en el apartado [4](#4-despliegue), que el despliegue se realice siempre dentro de una carpeta llamada *faas*. Eso es debido a que el broker elimina las imágenes mandando un mensaje *KILL* por el socket *zeromq* en tiempo de ejecución, pero cuando recibe una señal *SIGTERM* por parte de docker compose, termina a los workers utilizando *docker kill*, y debe conocer sus identificadores.

Para realizar los escalados se tienen que dar ciertas condiciones: El *scaleup* se activa cuando hay cinco o más trabajos en cola. Una vez se escala se dispara un timeout de 30 segundos, impidiendo que se realicen muchos escalados continuos quizás innecesarios. Una vez terminado este periodo podría volver a escalar si fuera necesario. En el caso del *scaledown*, su funcionamiento es diferente. Hay un intervalo que comprueba cada 5 segundos si la pool de workers no está vacía. En el caso de que en 5 comprobaciones consecutivas encuentre uno o más workers, un worker es eliminado y vuelve a cero el contador. 

#### 3.2.2.- Detalles de la implementación del worker

![worker](/md-images/worker-exec.png)
> Esquema a gran escala de una ejecución de un worker


El worker debe ser capaz de obtener las imágenes del registro y de ejecutarlas en base a los parámetros pasados. La comunicación con el worker se obtiene con el patrón *req-rep* de *zeromq*. El worker implementa el *req*, que se conecta a un *rep* en el broker previamente *bindeado* a un *socket*. Cuando un worker inicia su ejecución manda un mensaje vacío al broker, y éste le da un trabajo (si la cola de trabajos no está vacía) o lo añade a la pool de workers. Cuando el worker recibe un trabajo (mediante un *reply*) ejecuta la función en un contenedor docker hermano, al que bindea dos archivos: un *params.json* con los parametros de entrada y un *results.json* donde la función dejará los resultados, obteniendo así comunicación con el contenedor desplegado. Una vez obtenidos los resultados, los devolverá al *broker* con un *request*, que servirá a su vez para indicarle que está disponible de nuevo para ejecutar.

#### 3.2.3.- Script para la interacción con el sistema
    
Se ha desarrollado también un script *bash* para interactuar con el sistema. Implementa todas las funciones de la API del broker y ofrece una interfaz de comunicación útil. Para desplegar su menú de opciones ejecuta:

```bash
$ ./faas.sh help
```

## 4.- Despliegue

Existen dos maneras de desplegar el FaaS. Ambas hacen uso de la herramienta [Docker Compose](https://docs.docker.com/compose/) para lograrlo, por lo que será imprescindible disponer de esta herramienta en la máquina en la que se quiera desplegar. 

La primera consiste en crear las imágenes en local a partir de sus correspondientes Dockerfiles y el código fuente. Para ello será necesario clonar este repositorio y ejecutar el siguiente comando:

``` bash
$ docker-compose up --force-recreate --build
```

La opción *force-recreate* relanza los contenedores en caso de que ya existan, y la opción *build* vuelve a crear las imágenes. 

Esta opción de despliegue es útil si se quieren realizar cambios en el código, pues se aplicarían en cada lanzamiento, por lo que puede ser considerada como una opción de desarrollo.

La otra opción es hacer uso de las imágenes ya disponibles en dockerhub para realizar el despliegue. Estas imágenes son *jrodriguez96/brokerfaas* y *jrodriguez96/workerfaas*. Se haría uso de este dockerfile, disponible en *hub-deployment/faas/* :
```yaml
version: '2'
services:
    registry:
        restart: always
        image: registry:2
        ports:
            - 5000:5000
        volumes:
            - "faasRegistry:/var/lib/registry"
    broker:
        image: jrodriguez96/brokerfaas
        environment:
        - ZMQ_BIND_ADDRESS=tcp://*:2000
        ports:
        - 3333:3333
        volumes:
        - "/var/run/docker.sock:/var/run/docker.sock"
        links:
        - registry
    worker:
        image: jrodriguez96/workerfaas
        environment:
        - ZMQ_CONN_ADDRESS=tcp://broker:2000
        volumes:
        - "/var/run/docker.sock:/var/run/docker.sock"
        - "/tmp/requests:/worker/requestsworker"
        links:
        - broker
volumes:
  faasRegistry:
```

Para que este lanzamiento no falle, tiene que estar disponible el puerto 3333 en el host, pues es el que se utiliza en la imagen subida a *docker hub*. También es necesario que el *docker-compose.yml* se encuentre dentro de una carpeta llamada "faas", pues *Docker Compose* va a coger esa nomenclatura para darle nombre a los contenedores, y será utilizada por el *broker* para mandar señales.

## 5.- Pruebas

Se han realizado dos tipos de pruebas principalmente, las pruebas de validación y las de rendimiento.

### 5.1.- Pruebas de validación

Para llevar un control del funcionamiento y la correctitud del sistema, se proponen dos funciones, halladas en la carpeta */tests*. Estas dos funciones son *sum* y *dice*. 

La primera realiza la suma de dos valores introducidos, *a* y *b*. Un ejemplo de JSON válido para su llamada sería el siguiente: `'{"a":1,"b":2}'`. 

Por otro lado, la segunda es una simulación de una tirada de dados. Por defecto (si se llama sin argumentos) tira un dado de 6 caras. Acepta dos parámetros *num_dices*, que indica la cantidad de dados a lanzar y *size*, que indica el número de caras de cada dado lanzado. Devuelve la suma del valor de todos los dados así como una lista de los valores obtenidos en cada tirada. Un ejemplo de JSON válido para su llamada sería el siguiente: `'{"num_dices":100,"size":20}'`.

Estas dos funciones de ejemplo también han sido utilizadas para comprobar el funcionamiento del script de interacción con el sistema, `faas.sh`. Se ha comprobado que todos sus subcomandos funcionan correctamente.

### 5.2.- Pruebas de rendimiento

Para comprobar la eficiencia del sistema, se han reaizado una serie de pruebas de rendimiento. Estas pruebas consisten en realizar llamadas al sistema invocando la función *dice* sin parámetros, por lo que se el tiempo de ejecución real es despreciable, ya que lo único que realiza el contenedor de función levantado es una operación *math.random*. Estas llamadas a la función se van a realizar con un intervalo de tiempo, que podemos ver en la parte de arriba de cada figura. La línea *worker replicas* indica el número de réplicas activas en cada instante. Por otro lado, la línea *request time* indica el tiempo que ha tardado una petición desde que se mandó al sistema hasta que se recibió el resultado. Por último,la línea *exec time* indica el tiempo transcurrido desde que un worker da la orden de levantar el contenedor función hasta que éste finaliza su ejecución devolviendo el resultado.

Todas las pruebas se han realizado en un procesador *octa-core* en el sistema operativo *Ubuntu*.

#### Una petición por segundo

![plot 1 segundo](/plot/1sec.png)

> Se puede ver una gráfica muy clara del funcionamiento del sistema. Un worker no es suficiente para abastecer las peticiones sin evitar que crezca la cola, y por tanto su *request time*. Pero dos réplicas son demasiado, por lo que en el momento en el que la cola llega al *watermark* (puesto en 5 peticiones), se levanta un segundo worker, que baja el *request time* hasta igualarlo con el *exec time*, es decir, ninguna petición espera más de lo necesario. En el momento en el que se alcanza esta punto, se puede dar el caso en el que se dispare un *scale down*, pues se puede encontrar a alguno de los dos workers esperando. De ser así, se vuelve a una instancia y se repite el ciclo.

#### Dos peticiones por segundo

![plot 0.5 segundos](/plot/0-5sec.png)

> En este caso se puede observar como, al llegar más peticiones por segundo, se necesitan más réplicas. Cabe desctacar la pronunciada curva del *request time* en las primeras 100 peticiones. Esto es debido a que el sistema empieza parado, es decir, con un solo worker. Al no proporcionar una política de escalado instantáneo (como se comentaba en el apartado [3](#3.--implementación), al realizar un *scale up* es necesaria una espera hasta que se puede lanzar otra instancia), se tarda un tiempo en atender las peticiones y en llegar a un número de instancias que soporten ese *throughput*. En el momento que el sistema puede con la carga, el *request time* baja, como se puede ver alrededor de la petición 200, con 4 replicas. A partir de ahi se entra en este ciclo de *scale up* y *scale down*, similar al que se apreciaba en el ejemplo anterior.

#### Tres peticiones por segundo

![plot 0.33 segundos](/plot/0-33sec.png)

> Este caso es similar al anterior, pero magnificado. Es por eso que se ha realizado sobre 3000 peticiones y no sobre 1000, como el resto de ejemplos, pues no se podía apreciar correctamente el patrón obtenido. Se puede ver que el *request time* alcanza una máxima de alrededor de 35 segundos en su punto más elevado. A partir de ese punto, el tiempo baja hasta el mínimo sobre la petición 1000 y entra en el ciclo de escalado. Las instancias de workers oscilan entre 4 y 8 en este caso. Cabe destacar como aumenta el *exec time* a medida que se invocan más workers. Esto es debido a que todos los workers están en la misma máquina y utilizan el mismo hardware, por lo que éste se puede llegar a saturar.

#### Cuatro peticiones por segundo

![plot 0.25 segundos](/plot/0-25sec.png)

> En este caso, la gráfica difiere bastante del resto. Es un caso en el que no se llega a alcanzar esta oscilación estabilizada entre las réplicas y tiempo de petición. Se puede ver como el tiempo de petición crece a medida que llegan más peticiones, y como suben el número de instancias para intentar alcanzar el *throughput* necesario para ejecutar a un ritmo mayor a la llegada de peticiones. El factor determinante para saber si se alcanzará ese momento es el ratio al que crezca *exec time*. Si se pierde más tiempo en las ejecuciones que se gana añadiendo un worker nuevo, nunca se alcanzará el punto de equilibrio. En el caso de que esto se ejecutara en varios nodos, se puede suponer un *exec time* constante, por lo que, sea cual sea la frecuencia a la que entran las peticiones, siempre se podrían llegar a controlar.

## 6.- Trabajos Futuros

Se pueden proponer muchas mejoras a este trabajo, pues se ha llegado a una solución usable y funcional, pero hay bastante margen para alcanzar un sistema más completo. Entre estas mejoras se puede destacar:

* **Adaptar el sistema a un entorno multinodo** El uso de *docker-compose* ofrece una solución sencilla para un despliegue en una sola máquina, pero no es suficiente si se quiere desplegar en *hosts* remotos. 

* **Ofrecer una heurística de escalado más completa** Las políticas de escalado son funcionales, como se puede observar en las pruebas realizadas. No obstante, se pueden aplicar técnicas más inteligentes para optimizar aún más los recursos, como podría ser el uso de más *watermarks* a la hora de hacer *scaleup*, de modo que si la cola se satura mucho se responde antes.

* **Desplegar diferentes tipos de contenedores función** en base al tipo de suscripción del cliente. Es decir, ofrecer alternativas más o menos potentes con costes monetarios adaptados a ellas para poder dar un servicio más personalizado.

* **Mejorar las comunicaciones** Si bien es cierto que el patrón de comunicación *req-rep* entre el broker y los workers y el anónimato de estos últimos (el broker no los diferencia) suponen una solución elegante al problema, abriendo más tipos de canales, como pings, y enrutando los trabajos de manera concreta se puede obtener un control y una tolerancia a fallos mayor sobre el sistema.

## 7.- Conclusiones

Para concluir, el trabajo se puede considerar exitoso. Se ha diseñado e implementado un sistema funcional, que, pese a sus limitaciones, funciona. Se ha hecho uso de herramientas diversas, y se ha aprendido mucho de todas ellas. El sistema final es sencillo de utilizar y de desplegar, y se han obtenido medidas de rendimiento que soportan el trabajo realizado. Si bien se han quedado algunos puntos en el apartado [6](#6-trabajos-futuros) que habría sido interesante añadir, pero, por lo general, se ha alcanzado una buena solución.


