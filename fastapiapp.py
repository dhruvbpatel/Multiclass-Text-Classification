import streamlit as st
import pandas as pd

from sklearn.metrics import classification_report

import findspark
findspark.init()

import sparknlp

spark = sparknlp.start()


import pyspark

from pyspark.ml import Pipeline

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.sql.types import StringType


import warnings
warnings.filterwarnings('ignore')



## api app
from fastapi import FastAPI
from pydantic import BaseModel

# from 
# sc.setLogLevel("ERROR")


# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# sc = SparkContext.getOrCreate()
# spark = SparkSession(sc)

class Input(BaseModel):
	text:str

app=FastAPI()



def predict(text_input):
	


	document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

	use = UniversalSentenceEncoder.pretrained() \
	 .setInputCols(["document"])\
	 .setOutputCol("sentence_embeddings")

	sentimentdl = SentimentDLModel.load("./nbs/tmp_sentimentdl_model") \
	  .setInputCols(["sentence_embeddings"])\
	  .setOutputCol("class")

	pipeline = Pipeline(`
	    stages = [
	        document,
	        use,
	        sentimentdl
	    ])

	def test_sentiment_func(test_input):
	    
	    test_sentence_df = spark.createDataFrame([test_input],StringType()).toDF("text")
	    test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
	    test_ans_df=test_pred.select("class.result").toPandas()
	    # st.write(test_ans_df)
	    return test_ans_df

	if text_input is not "":
		ans=test_sentiment_func(text_input)
		return ans


@app.put("/predict")
def main(d.Input):
	return predict()



if __name__ == '__main__':
    main()
