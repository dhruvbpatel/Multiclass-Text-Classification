
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
	option = ["Sentiment Classification", "News Classification"]
	choice = st.sidebar.selectbox("Select Task",option)

	if choice=="Sentiment Classification":


		
		st.header("Sentiment Classification")

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

		if st.checkbox("Predict on CSV file",False):

			st.set_option('deprecation.showfileUploaderEncoding', False)  ## ignore warning

			data = st.file_uploader("Upload Data here (CSV file only)", type=['csv'])
			if data is not None:
				df = pd.read_csv(data)
				# col_list = df.columns.tolist()
				
				# st.subheader("RAW DATA")
				# st.write(df)


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

	## multiclass text classifier

	if choice=="News Classification":

		
		st.header("News Text Classification")	

		html_header = """
		<head>
	    <title>News Text Classification</title>
	    </head>
	    <div style ="background-color:#00ACEE;padding:10px">
	    <h2 style="color:white;text-align:center;">News Text Classification</h2>
	    </div>
		"""
		st.markdown(html_header,unsafe_allow_html=True)
		text_input=""
		text_input = st.text_input("Enter your text here")

		document = DocumentAssembler()\
	    .setInputCol("description")\
	    .setOutputCol("document")

		use = UniversalSentenceEncoder.pretrained() \
		 .setInputCols(["document"])\
		 .setOutputCol("sentence_embeddings")

		classsifierdl = ClassifierDLModel.load("./nbs/tmp_classifierDL_model") \
		  .setInputCols(["sentence_embeddings"])\
		  .setOutputCol("class")

		pipeline = Pipeline(
		    stages = [
		        document,
		        use,
		        classsifierdl
		    ])

		def test_classification_func(test_input):
		    
		    test_sentence_df = spark.createDataFrame([test_input],StringType()).toDF("description")
		    test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
		    test_ans_df=test_pred.select("class.result").toPandas()
		    st.write(test_ans_df)

		if text_input is not "":
			test_classification_func(text_input)




	hide_streamlit_style = """ 
		<style>
				title{visibity:hidden;}
		</style>
		"""
	st.markdown(hide_streamlit_style,unsafe_allow_html=True)




if __name__ == '__main__':
    main()
