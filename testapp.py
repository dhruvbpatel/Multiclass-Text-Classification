import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from sklearn.metrics import classification_report

import findspark
findspark.init()

import sparknlp
spark = sparknlp.start(gpu=True)


import pyspark
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.pretrained import PretrainedPipeline

import json
import time
import warnings
import os

warnings.filterwarnings("ignore")


# from
# sc.setLogLevel("ERROR")


# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# sc = SparkContext.getOrCreate()
# spark = SparkSession(sc)


def data_loader():
    st.set_option("deprecation.showfileUploaderEncoding", False)  ## ignore warning

    data = st.file_uploader("Upload Data here (CSV file only)", type=["csv"])

    if data is not None:
        pass
        
    return data


def make_pipeline():
    document = DocumentAssembler().setInputCol("text").setOutputCol("document")

    use = (
        UniversalSentenceEncoder.pretrained()
        .setInputCols(["document"])
        .setOutputCol("sentence_embeddings")
    )

    sentimentdl = (
        SentimentDLModel.load("./nbs/tmp_sentimentdl_model")
        .setInputCols(["sentence_embeddings"])
        .setOutputCol("class")
    )

    pipeline = Pipeline(stages=[document, use, sentimentdl])
    return pipeline


def test_sentiment_func(test_input, pipeline):

    test_sentence_df = spark.createDataFrame([test_input], StringType()).toDF("text")
    test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
    test_ans_df = test_pred.select("class.result").toPandas()
    st.subheader("Predicted Sentiment: ")
    st.write(test_ans_df)


def make_news_pipeline():
    document = DocumentAssembler().setInputCol("description").setOutputCol("document")

    use = (
        UniversalSentenceEncoder.pretrained()
        .setInputCols(["document"])
        .setOutputCol("sentence_embeddings")
    )

    classsifierdl = (
        ClassifierDLModel.load("./nbs/tmp_classifierDL_model")
        .setInputCols(["sentence_embeddings"])
        .setOutputCol("class")
    )

    pipeline = Pipeline(stages=[document, use, classsifierdl])

    return pipeline


def test_news_classification_func(test_input, pipeline):

    test_sentence_df = spark.createDataFrame([test_input], StringType()).toDF(
        "description"
    )
    test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
    test_ans_df = test_pred.select("class.result").toPandas()
    st.write(test_ans_df)

def make_fake_news_pipeline():
	model_name = 'classifierdl_use_fakenews'
	documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

	use = UniversalSentenceEncoder.pretrained(lang="en") \
	 .setInputCols(["document"])\
	 .setOutputCol("sentence_embeddings")

	document_classifier = ClassifierDLModel.pretrained(model_name)\
	  .setInputCols(['document', 'sentence_embeddings']).setOutputCol("class")

	nlpPipeline = Pipeline(stages=[
	 documentAssembler, 
	 use,
	 document_classifier
	 ])
	return nlpPipeline

def test_fake_news_classification_func(test_input,pipeline):


    # test_sentence_df = spark.createDataFrame([test_input], StringType()).toDF("text")
    # test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
    # test_ans_df = test_pred.select("class.result").toPandas()
    # st.subheader("Predicted Sentiment: ")
    # st.write(test_ans_df)



	empty_df = spark.createDataFrame([['']]).toDF("text")
	pipelineModel = pipeline.fit(empty_df)
	# df = spark.createDataFrame(pd.DataFrame({"text":str(test_input)}))
	df = spark.createDataFrame([test_input],StringType()).toDF("text")
	result = pipelineModel.transform(df)
	ans = result.toPandas()
	st.subheader("Prediction")
	st.write(ans['class'][0][0][3])




def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'


def make_plots(test_ans_df):
	plot_df = test_ans_df['result'].apply(lambda x : x[0])


	plt.figure(figsize=(6,7))
	splot = sns.countplot(plot_df)
	plt.plot(size=(4,7))
	for p in splot.patches:
		splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
	st.pyplot()


### CSV Prediction functions

def predict_csv_sentiment(pipeline, test_input):
    st.subheader("CSV Data: ")
    st.dataframe(test_input)
    test_sentence_df = spark.createDataFrame(test_input)
    # st.write(type(test_sentence_df))

    test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
    test_ans_df = test_pred.select("text", "class.result").toPandas()
    st.subheader("Predicted Sentiment: ")
    st.write(test_ans_df)
    make_plots(test_ans_df)


def predict_csv_classification(pipeline,test_input):
	st.subheader("CSV Data: ")
	st.dataframe(test_input)
	test_sentence_df = spark.createDataFrame(test_input)
	test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
	test_ans_df = test_pred.select("description","class.result").toPandas()
	st.subheader("Predicted Sentiment: ")
	st.write(test_ans_df)
	make_plots(test_ans_df)

def main():

    first_time_load_sentiment = True

    st.sidebar.title("Select Task")
    option = ["Sentiment Classification", "News Text Classification","Fake News Classification"]
    choice = st.sidebar.selectbox("Select Task", option)

    if choice == "Sentiment Classification":

        st.header("Sentiment Classification")

        html_header = """
		<head>
	    <title>Sentiment Classification</title>
	    </head>
	    <div style ="background-color:#00ACEE;padding:10px">
	    <h2 style="color:white;text-align:center;">Sentiment Classification</h2>
	    </div>
		"""
        st.markdown(html_header, unsafe_allow_html=True)
        text_input = st.text_input("Enter your text here")

        if st.checkbox("Predict on CSV file", False):


        	markdown_text = """
        	<div>
	    	<h2 style="color:red;;font-size:20px">
	    	***The CSV file should be in following format:***
	    	<center>
	    	<br>
	    	<ol>
	    	<li style="color:black;text-align:left;font-size:10px">In CSV file text header column name should be:"text"</li>
	    	<li style="color:black;text-align:left;font-size:10px">Only a single column with text values should be there in CSV file</li>
	    	</ol>
	    	</center>
	    	</h2>
	    	</div>


        	"""
        	st.markdown(markdown_text,unsafe_allow_html=True)

        	data = data_loader()
        	if data is not None:
	        	df_pdf = pd.read_csv(data)
	        	pipeline = make_pipeline()
	        	predict_csv_sentiment(pipeline,df_pdf)

        if text_input is not "":
            # start = time.time()
            if first_time_load_sentiment == True:
                pipeline = make_pipeline()
                test_sentiment_func(text_input, pipeline)
                first_time_load_sentiment = False
            else:
                test_sentiment_func(text_input, pipeline)
            # end = time.time()
            # st.write(end-start)

    ## multiclass text classifier

    if choice == "News Text Classification":

        st.header("News Text Classification")

        html_header = """
		<head>
	    <title>News Text Classification</title>
	    </head>
	    <div style ="background-color:#6b7380;padding:10px">
	    <h2 style="color:white;text-align:center;">News Text Classification</h2>
	    </div>
		"""

        ### input text

        st.markdown(html_header, unsafe_allow_html=True)
        text_input = ""
        text_input = st.text_input("Enter your text here")

        if st.checkbox("Predict on CSV file", False):


        	markdown_text = """
        	<div>
	    	<h2 style="color:red;;font-size:20px">
	    	***The CSV file should be in following format:***
	    	<center>
	    	<br>
	    	<ol>
	    	<li style="color:black;text-align:left;font-size:10px">In CSV file text header column name should be:"text"</li>
	    	<li style="color:black;text-align:left;font-size:10px">Only a single column with text values should be there in CSV file</li>
	    	</ol>
	    	</center>
	    	</h2>
	    	</div>


        	"""
        	st.markdown(markdown_text,unsafe_allow_html=True)

        	data = data_loader()
        	if data is not None:
	        	df_pdf = pd.read_csv(data)
	        	pipeline = make_news_pipeline()
	        	predict_csv_classification(pipeline,df_pdf)

        if text_input is not "":
            # start = time.time()
            if first_time_load_sentiment == True:
                pipeline = make_pipeline()
                test_sentiment_func(text_input, pipeline)
                first_time_load_sentiment = False
            else:
                test_sentiment_func(text_input, pipeline)






        first_time_load = True

        if text_input is not "":
            if first_time_load == True:
                news_pipeline = make_news_pipeline()
                test_news_classification_func(text_input, news_pipeline)
                first_time_load = False
            else:
                test_news_classification_func(text_input, news_pipeline)

### FAKE NEWS CLASSIFIER ###
    if choice == option[2]:
    	st.header("Fake News Classification")
    	html_header= """
    	<head>
	    <title>Fake News Classification</title>
	    </head>
	    <div style ="background-color:red;padding:10px">
	    <h2 style="color:white;text-align:center;">News Text Classification</h2>
	    </div>

    	"""
    	st.markdown(html_header,unsafe_allow_html=True)
    	text_input=""
    	text_input=st.text_input("Enter your text here")

    	if text_input is not "":
            # start = time.time()
            if first_time_load_sentiment == True:
                pipeline = make_fake_news_pipeline()
                test_fake_news_classification_func(text_input, pipeline)
                first_time_load_sentiment = False
            else:
                test_fake_news_classification_func(text_input,pipeline)




    hide_streamlit_style = """ 
		<style>
				title{visibity:hidden;}
		</style>
		"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
