# 0 Para cada tienda, obtener la transacción de máximo importe.

sc.textFile("/datasets/purchases/purchases.txt").map(lambda s: s.split("\t")).map(lambda rec: (rec[2], float(rec[4]))).reduceByKey(max).take(1000)


# 1 Suma total de ventas para cada categoría de producto.

sc.textFile("/datasets/purchases/purchases.txt").map(lambda s: s.split("\t")).map(lambda rec: (rec[3], float(rec[4]))).reduceByKey(lambda x,y:x+y ).take(1000)


# 2 Número total de accesos al recurso "/assets/img/home-logo.png”

sc.textFile("/datasets/accesslog/access_log").map(lambda s: s.split(" ")).map(lambda x: (x[6],1)).filter(lambda rec: rec[0]=="/assets/img/home-logo.png").count()


# 3 Número total de accesos desde la misma dirección IP: 10.223.157.186

sc.textFile("/datasets/accesslog/access_log").map(lambda s: s.split(" ")).map(lambda rec: (rec[0],1)).filter(lambda rec: rec[0]=="10.223.157.186").count()

# 4 Recurso web con mayor número de accesos

sc.textFile("/datasets/accesslog/access_log").map(lambda s: s.split(" ")).map(lambda rec: (rec[6],1)).reduceByKey(lambda x,y: x+y).max(key=lambda x: x[1])
