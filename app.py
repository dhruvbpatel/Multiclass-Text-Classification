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


# from 
# sc.setLogLevel("ERROR")


# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# sc = SparkContext.getOrCreate()
# spark = SparkSession(sc)

def main():
	st.sidebar.title("Select Task")
	st.header("Sentiment Classification")

	option = ["Sentiment Classification", "News Classification"]
	st.sidebar.selectbox("Select Task",option)

	html_header = """
	<head>
    <title>Sentiment Classification</title>
    </head>
    <div style ="background-color:#00ACEE;padding:10px">
    <h2 style="color:white;text-align:center;">Sentiment Classification</h2>
    </div>
	"""
	st.markdown(html_header,unsafe_allow_html=True)
	text_input = st.text_input("Enter your text here")

	document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

	use = UniversalSentenceEncoder.pretrained() \
	 .setInputCols(["document"])\
	 .setOutputCol("sentence_embeddings")

	sentimentdl = SentimentDLModel.load("./nbs/tmp_sentimentdl_model") \
	  .setInputCols(["sentence_embeddings"])\
	  .setOutputCol("class")

	pipeline = Pipeline(
	    stages = [
	        document,
	        use,
	        sentimentdl
	    ])

	def test_sentiment_func(test_input):
	    
	    test_sentence_df = spark.createDataFrame([test_input],StringType()).toDF("text")
	    test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
	    test_ans_df=test_pred.select("class.result").toPandas()
	    st.write(test_ans_df)

	if text_input is not "":
		test_sentiment_func(text_input)





	hide_streamlit_style = """ 
		<style>
				title{visibity:hidden;}
		</style>
		"""
	st.markdown(hide_streamlit_style,unsafe_allow_html=True)




if __name__ == '__main__':
    main()
