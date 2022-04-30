import os
import pandas as pd
import openpyxl

cwd = os.getcwd()

# Filepath to stock and tweet data
stock_data_path = cwd + "/HistoricalData_1649560561913.csv"
tweet_data_folder = cwd + "/MuskTwitterData"

# 0 if data isn't prepped and need to prep, 1 if it is
dataPrepped = 1
tweetsCompiled = 1
tweetsCombined = 0

def compileTweets():
    # Dataframes containing data for each year
    tweets_2022 = pd.read_csv(tweet_data_folder + "/2022.csv")
    tweets_2021 = pd.read_csv(tweet_data_folder + "/2021.csv")
    tweets_Historical = pd.read_csv(tweet_data_folder + "/2020.csv")

    # Combine dataframes and write to excel file
    # 2022 and 2021 dataframes slightly different format, concatenate separately
    tweets_LastTwoYears = pd.concat([tweets_2022, tweets_2021])
    tweets_LastTwoYears = tweets_LastTwoYears[['date', 'replies_count', 'retweets_count', 'likes_count']]
    tweets_Historical = tweets_Historical[['date', 'nlikes', 'nreplies', 'nretweets']]
    # tweets_YearsBefore = pd.concat([tweets_2020, tweets_2019, tweets_2018, tweets_2017])
    tweets_Historical = tweets_Historical.rename(columns={'nlikes': 'likes_count', 'nreplies': 'replies_count',
                                                          'nretweets': 'retweets_count'})

    allTweets = pd.concat([tweets_LastTwoYears, tweets_Historical])
    allTweets.to_excel(tweet_data_folder + "/tweets_Historic.xlsx")

    pass

def combineTweets():
    allTweets = pd.read_excel(tweet_data_folder + "/tweets_Historic.xlsx")
    allTweets = allTweets.drop(columns=['Unnamed: 0'])
    combinedTweets = pd.DataFrame(columns=['date', 'numTweets', 'replies_count', 'replies_average', 'retweets_count',
                                           'retweets_average', 'likes_count', 'likes_average'])

    # Loop through the dataframe and consolidate num tweets, retweets, likes for each date
    length = allTweets.shape[0]
    compareDate = allTweets.iat[0, 0]
    compareDate = compareDate[0:10]
    numTweets = 0
    numReplies = 0
    numRetweets = 0
    numLikes = 0
    for row in range(length):
        date = allTweets.iat[row, 0]
        date = date[0:10]
        if date == compareDate:
            numTweets += 1
            numReplies += allTweets.iat[row, 1]
            numRetweets += allTweets.iat[row, 2]
            numLikes += allTweets.iat[row, 3]
        else:
            # Append new row
            newRow = {'date': compareDate, 'numTweets': numTweets, 'replies_count': numReplies,
                      'replies_average': numReplies/numTweets, 'retweets_count': numRetweets,
                      'retweets_average': numRetweets/numTweets, 'likes_count': numLikes,
                      'likes_average': numLikes/numTweets}
            combinedTweets = combinedTweets.append(newRow, ignore_index=True)
            # Update comparators and indexes
            compareDate = date
            numTweets = 1
            numReplies = allTweets.iat[row, 1]
            numRetweets = allTweets.iat[row, 2]
            numLikes = allTweets.iat[row, 3]
    
    combinedTweets.to_excel(tweet_data_folder + "/tweets_Prepared.xlsx")


def prep_data():
    # Read Tesla Historical data CSV

    priceHist = pd.read_csv(stock_data_path)

    pass

if tweetsCompiled == 0:
    compileTweets()

if dataPrepped == 0:
    prep_data()

if tweetsCombined == 0:
    combineTweets()
