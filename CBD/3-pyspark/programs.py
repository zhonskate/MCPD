# 0 Para cada tienda, obtener la transacción de máximo importe.

sc.textFile("/datasets/purchases/purchases.txt").map(lambda s: s.split("\t")).map(lambda rec: (rec[2], float(rec[4]))).reduceByKey(max).take(1000)


# 1 Suma total de ventas para cada categoría de producto.

sc.textFile("/datasets/purchases/purchases.txt").map(lambda s: s.split("\t")).map(lambda rec: (rec[3], float(rec[4]))).reduceByKey(lambda x,y:x+y ).take(1000)


# 2 Número total de accesos al recurso "/assets/img/home-logo.png”


