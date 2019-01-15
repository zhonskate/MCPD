# Diseño e implementación de un FaaS

## Índice

[1.- Introducción](#1.--Introducción)

[2.- Diseño](#2.--diseño)

[3.- Implementación](#3.--implementación)

[4.- Despliegue](#4.--despliegue)

[5.- Pruebas](#5.--pruebas)

[6.- Trabajos Futuros](#6.--trabajos-futuros)

[7.- Conclusiones](#7.--conclusiones)

## 1.- Introducción

// investigar

## 2.- Diseño

// diagramas, etc

## 3.- Implementación

// tecnologias utilizadas

//insights

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

// hablar de los módulos programados. Pruebas de tiempo?

## 6.- Trabajos Futuros

## 7.- Conclusiones


