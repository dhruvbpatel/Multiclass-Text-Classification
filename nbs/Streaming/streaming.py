import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext(appName="TweeSent0")
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 5)   #Streaming will execute in each 3 seconds
lines = ssc.socketTextStream(hostname = "localhost", port = 10000)  #'log/ mean directory name
counts = lines.flatMap(lambda line: line.split(" ")) \
    .map(lambda x: (x, 1)) \
    .reduceByKey(lambda a, b: a + b)
counts.pprint()
ssc.start()
ssc.awaitTermination()
