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

import time
import warnings

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
        # df_pd = pd.DataFrame(data)
        # sparkDF = spark.createDataFrame(df_pd)
        # st.write(type(sparkDF))
        # st.dataframe(data)
        # col_list = df.columns.tolist()

        # st.subheader("RAW DATA")
        # st.write(df)

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


def predict_csv_sentiment(pipeline, test_input):
	st.dataframe(test_input)

	test_sentence_df = spark.createDataFrame(test_input)
	# st.write(type(test_sentence_df))

	test_pred = pipeline.fit(test_sentence_df).transform(test_sentence_df)
	test_ans_df = test_pred.select("label","text","class.result").toPandas()
	st.subheader("Predicted Sentiment: ")
	st.write(test_ans_df)


def main():

    first_time_load_sentiment = True

    st.sidebar.title("Select Task")
    option = ["Sentiment Classification", "News Classification"]
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

            data = data_loader()
            # st.write(type(data))
            df_pd = pd.read_csv(data)
            # sparkDF = spark.createDataFrame([df_pd],StringType()).toDF("text")


            pipeline = make_pipeline()
            # st.write(type(df_pd))
            predict_csv_sentiment(pipeline,df_pd)

                      

            # st.write(preds.select('text','label','class.result').show(50,truncate=50))

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

    if choice == "News Classification":

        st.header("News Text Classification")

        html_header = """
		<head>
	    <title>News Text Classification</title>
	    </head>
	    <div style ="background-color:#00ACEE;padding:10px">
	    <h2 style="color:white;text-align:center;">News Text Classification</h2>
	    </div>
		"""

        ### input text

        st.markdown(html_header, unsafe_allow_html=True)
        text_input = ""
        text_input = st.text_input("Enter your text here")

        first_time_load = True

        if text_input is not "":
            if first_time_load == True:
                news_pipeline = make_news_pipeline()
                test_news_classification_func(text_input, news_pipeline)
                first_time_load = False
            else:
                test_news_classification_func(text_input, news_pipeline)

    hide_streamlit_style = """ 
		<style>
				title{visibity:hidden;}
		</style>
		"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
