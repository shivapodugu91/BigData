# Databricks notebook source
#Book-Crossing dataset
display(dbutils.fs.ls('/FileStore/tables/jcroftps1499817600813'))

# COMMAND ----------

import sys
import os
from test_helper import Test

baseDir = os.path.join('FileStore')
inputPath = os.path.join('tables', '3l60imup1499819875788')

bookRatingsFilename = os.path.join(baseDir, inputPath, 'BX_Book_Ratings-05a61.csv')
booksFilename = os.path.join(baseDir, inputPath, 'BX_Books-c72a2.csv')
usersFilename=os.path.join(baseDir, inputPath, 'BX_Users-53089.csv')

# COMMAND ----------

#numPartitions = 2
rawBookRatings = sc.textFile(bookRatingsFilename)#.repartition(numPartitions)
rawBooks = sc.textFile(booksFilename)
rawUsers = sc.textFile(usersFilename)

# COMMAND ----------

#User-ID;"ISBN";"Book-Rating"
def get_rawBookRatings_tuple(entry):
   
    items = entry.split(';')
    items[2]=items[2].replace('"','')
  
    return int(items[0].replace('"','')), (items[1].replace('"','')), int(items[2].replace(',',''))


# COMMAND ----------

#ISBN;"Book-Title";"Book-Author"
def get_rawbooks_tuple(entry):
    
    items = entry.split(';')
    return (items[0].replace('"','')), (items[1].replace('"','')), (items[2].replace('"',''))

# COMMAND ----------

#User-ID;"Location";"Age"
def get_rawUsers_tuple(entry):
    
    items = entry.split(';')
    return (items[0]), (items[1]), (items[2])

# COMMAND ----------


bookRatingsRDD = rawBookRatings.map(get_rawBookRatings_tuple).cache()
booksRDD = rawBooks.map(get_rawbooks_tuple).cache()



# COMMAND ----------

bookRatingsRDDCount=bookRatingsRDD.count()
booksRDDCount = booksRDD.count()


# COMMAND ----------

print 'There are %s ratings and %s books in the datasets' % (bookRatingsRDDCount, booksRDDCount)
print 'Ratings: %s' % bookRatingsRDD.take(40)
print 'books: %s' % booksRDD.take(40)

# COMMAND ----------

def sortFunction(tuple):
    """ Construct the sort string (does not perform actual sorting)
    Args:
        tuple: 
    Returns:
        sortString: the value to sort with, 'rating MovieName'
    """
    key = unicode('%.3f' % tuple[0])
    value = tuple[1]
    return (key + ' ' + value)

# COMMAND ----------

def getCountsAndAverages(IDandRatingsTuple):
    """ Calculate average rating
    Args:
    #User-ID;"ISBN";"Book-Rating"
        IDandRatingsTuple: a single tuple of (ISBN, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (ISBN, (number of ratings, averageRating))
    """
    
    ISBN=IDandRatingsTuple[0]
    ratinglist=(IDandRatingsTuple[1])
    NumberOfRatings=len(ratinglist)
    sum=0
    for r in ratinglist:
      sum=sum+r
    avg=float(sum)/NumberOfRatings
    tup1=(ISBN,(NumberOfRatings,avg))
    
    return  tup1

# COMMAND ----------

# From bookRatingsRDD with tuples of (UserID, ISBN, Rating) create an RDD with tuples of
# the (ISBN, iterable of Ratings for that ISBN)
bookIDsWithRatingsRDD = (bookRatingsRDD.map(lambda x:(x[1],x[2])).groupByKey())
print 'bookIDsWithRatingsRDD: %s\n' % bookIDsWithRatingsRDD.take(3)


# COMMAND ----------

# Using `booksWithRatingsRDD`, compute the number of ratings and average rating for each book to
# yield tuples of the form (ISBN, (number of ratings, average rating))

bookIDsWithAvgRatingsRDD = bookIDsWithRatingsRDD.map(lambda r: getCountsAndAverages(r))
print 'bookIDsWithAvgRatingsRDD: %s\n' % bookIDsWithAvgRatingsRDD.take(30)

# COMMAND ----------

# To `bookIDsWithAvgRatingsRDD`, apply RDD transformations that use `booksRDDCount` to get the movie
# names for `movieIDsWithAvgRatingsRDD`, yielding tuples of the form
# (average rating, book name, number of ratings)

bookNameWithAvgRatingsRDD=(booksRDD.join(bookIDsWithAvgRatingsRDD).map(lambda x: (x[1][1][1],x[1][0],x[1][1][0])))

# COMMAND ----------

print 'bookNameWithAvgRatingsRDD: %s\n' % bookNameWithAvgRatingsRDD.take(20)

# COMMAND ----------

bookLimitedAndSortedByRatingRDD = (bookNameWithAvgRatingsRDD
                                    #.filter(lambda x:(x[2]>500))
                                    .sortBy(sortFunction, False))
print 'books with highest ratings: %s' % bookLimitedAndSortedByRatingRDD.take(10)

# COMMAND ----------

#newBookRatingsRDD = bookRatingsRDD.zipWithIndex()
newBookRatingsRDD= bookRatingsRDD
#newBookRatingsRDD=newBookRatingsRDD.collectAsMap()#.map(lambda x:(x[1],x[0][1],x[0][2],x[0][3]))
print bookRatingsRDD.take(2)

# COMMAND ----------

distinctISBNs = bookRatingsRDD.map(lambda x : (x[1])).distinct().zipWithIndex().collectAsMap()

print distinctISBNs

# COMMAND ----------

print bookRatingsRDD.map(lambda x : (x[1])).count()
print bookRatingsRDD.map(lambda x : (x[1])).distinct().count()
print distinctISBNs.count()

# COMMAND ----------

newBookRatingsRDD=bookRatingsRDD.map(lambda x:(x[0],int(distinctISBNs[x[1]]),x[2]))

# COMMAND ----------

print  newBookRatingsRDD.take(2)

# COMMAND ----------

#bookRatingsWithUserIdISBN = BookRatingsRDD.map(lambda x :(distinctUsers[x[0]] ,distinctISBNs[x[1]],float(x[2])))

# COMMAND ----------



trainingRDD, testRDD = newBookRatingsRDD.randomSplit([8, 2], seed=0L)

print 'Training: %s, test: %s\n' % (trainingRDD.count(),testRDD.count())

print trainingRDD.take(3)
print testRDD.take(3)

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
import math
def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each book and each user where each entry is in the form
                      (UserID, bookID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, bookID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, bookId), Rating)
  
    predictedReformattedRDD = predictedRDD.map(lambda x:((x[0],x[1]),x[2]))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    
    actualReformattedRDD = actualRDD.map(lambda x:((x[0],x[1]),x[2]))
      
    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect()
    #squaredErrorsRDD = (predictedReformattedRDD.<FILL IN>)
    squaredErrorsRDD = (predictedReformattedRDD.join(actualReformattedRDD).map(lambda((x,y),z):z).map(lambda (x,y):(x-y)**2))
    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.sum()

    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
 
    return (math.sqrt(float(totalError)/numRatings))

# COMMAND ----------

## training
#print trainingRDD.take(3)
#print testRDD.take(3)
from pyspark.mllib.recommendation import ALS

TestForPredictRDD = testRDD.map(lambda x :(x[0],x[1]))
seed = 30L
iterations = 8
regularizationParameter = 0.99
ranks = [17]
errors = [0]
err = 0
tolerance = 0.03

minError = float('inf')
bestRank = -1
bestIteration = -1
for rank in ranks:
    model = ALS.train(trainingRDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularizationParameter)
    predictedRatingsRDD = model.predictAll(TestForPredictRDD)
    error = computeError(predictedRatingsRDD, testRDD)
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < minError:
        minError = error
        bestRank = rank

print 'The best model was trained with rank %s' % bestRank

# COMMAND ----------


