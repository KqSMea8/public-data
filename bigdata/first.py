#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
spark-submit --master local[4] first.py 
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql import Row

from os.path import expanduser, join, abspath

 


appName = "hello spark"
master = "local"
spark = SparkSession.builder.appName(appName).master(master).getOrCreate()


def test():
    logFile = "/Users/gaotianpu/Bigdata/spark-2.2.0-bin-hadoop2.7/README.md"  # Should be some file on your system
    logData = spark.read.text(logFile).cache()

    numAs = logData.filter(logData.value.contains('a')).count()
    numBs = logData.filter(logData.value.contains('b')).count()

    print("Lines with a: %i, lines with b: %i" % (numAs, numBs))


def test2(): 

    spark.sql("""CREATE TABLE IF NOT EXISTS feed_user_behavior
(
    event_date INT COMMENT "日期",
    nid  string COMMENT "文章id",
    list_send  FLOAT COMMENT "下发",
    list_show FLOAT COMMENT "展现",
    list_click FLOAT COMMENT "点击",
    list_dislike FLOAT COMMENT "不喜欢",
    list_show_topic FLOAT COMMENT "主题队列-展现",
    list_click_topic FLOAT COMMENT "主题队列-点击",
    details_like FLOAT COMMENT "点赞",
    details_dislike FLOAT COMMENT "点踩",
    details_show FLOAT COMMENT "展现",
    details_click_comments FLOAT COMMENT "点击评论",
    details_click_favorite FLOAT COMMENT "收藏",
    details_click_share FLOAT COMMENT "分享",
    details_click_notes FLOAT COMMENT "点击说明文字",
    details_click_comment_user FLOAT COMMENT "点击评论用户",
    details_comments FLOAT COMMENT "评论",
    details_bad_comments FLOAT COMMENT "负面评论",
    details_like_comments FLOAT COMMENT "给评论点赞",
    details_author_follows FLOAT COMMENT "关注",
    details_read_time FLOAT COMMENT "阅读时长",
    --details_read_frequency FLOAT COMMENT "阅读次数",
    ext_1 FLOAT COMMENT "阅读时长-65",
    ext_2 FLOAT COMMENT "阅读次数-65",
    ext_3 FLOAT COMMENT "阅读时长-346",
    ext_4 FLOAT COMMENT "阅读次数-346",
    ext_5 FLOAT COMMENT "",
    ext_6 FLOAT COMMENT "",
    ext_7 FLOAT COMMENT "",
    ext_8 FLOAT COMMENT "",
    ext_9 FLOAT COMMENT "",
    ts  INT COMMENT "后验数据产出时间戳"
)""")
    df = spark.sql("LOAD DATA LOCAL INPATH '/Users/gaotianpu/Downloads/doc.daily.20171106.csv' INTO TABLE feed_user_behavior")


    # schema = StructType([
    #     StructField("event_date", IntegerType(), True),
    #     StructField("nid", IntegerType(), True),
    #     StructField("list_send", IntegerType(), True)
    # ]) 

    # df = spark.read.csv(
    #     '/Users/gaotianpu/Downloads/doc.daily.20171106.csv',
    #     header='false', 
    #     inferSchema='true')
        
    # print df.count()  #249999
    # df.createOrReplaceTempView("test")
    sqlDF = spark.sql("SELECT * FROM feed_user_behavior limit 10")

    print df.columns

    for x in sqlDF.collect():
        print x['_c1']

    # print sqlDF['_c1']
    #WARN Utils: Truncated the string representation of a plan since it was too large.
    #This behavior can be adjusted by setting 'spark.debug.maxToStringFields' in SparkEnv.conf

##if __name__ == "___main__": #不能加这个？

# test()


test2()
spark.stop()
