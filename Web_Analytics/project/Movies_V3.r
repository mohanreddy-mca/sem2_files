###################Import Libraries #########################
library(arules)
library(dplyr)
library(reshape2)
library(Matrix)
library(stringr)
library(stringdist)
library(ggplot2)
#############################################################

#######################################################################
# the supporting functions
#######################################################################

#remove duplicate items from a basket (itemstrg)
uniqueitems <- function(itemstrg) {
  unique(as.list(strsplit(gsub(" ","",itemstrg),","))[[1]])
}

# execute ruleset using movie_id as rule antecedent (handles single item antecedents only)
makepreds <- function(movie_id, rulesDF) {
  antecedent = paste("{",movie_id,"} =>",sep="") 
  #print(antecedent)
  firingrules = rulesDF[grep(antecedent, rulesDF$rules,fixed=TRUE),1]
  #print(firingrules)
  gsub(" ","",toString(sub("\\}","",sub(".*=> \\{","",firingrules))))
}

# count how many predictions are in the basket of movie_id already seen by that user 
# Caution : refers to "baskets" as a global
checkpreds <- function(preds, user_id) {
  plist = preds[[1]]
  blist = baskets[baskets$user_id == user_id,"movie_id"][[1]]
  cnt = 0 
  for (p in plist) {
    if (p %in% blist) cnt = cnt+1
  }
  cnt
}

# count all predictions made
countpreds <- function(predlist) {
  len = length(predlist)
  if (len > 0 && (predlist[[1]] == "")) 0 # avoid counting an empty list
  else len
}

####################################End Functions ###############################

#Construct Association Rules from Rating Data
setwd("C:\\Classes\\Sem2_jul_to_dec_2017\\WebAnalytics\\Project1\\V2\\")


#################### Read data and Split data #####################################
ratings_100k = read.csv("ratings_100k.csv",
                        colClasses = c("integer","integer","integer","integer"),
                        sep=",",
                        stringsAsFactors = FALSE)

movies_100k = read.csv("movies_100k.csv",
                       colClasses = c("integer","character",NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),
                       sep=",",
                       stringsAsFactors = FALSE)

users_100k = read.csv("users_100k.csv",
                      colClasses = c("integer","integer","character","character","character"),
                      sep=",",
                      stringsAsFactors = FALSE)


#train_data_100k = subset(ratings_100k, user_id < 600)
#write.csv(train_data_100k, file = "train_data_100k.csv") 
#test_data_100k = subset(ratings_100k, user_id >= 600)
#write.csv(test_data_100k, file = "test_data_100k.csv") 


####################################################################################

ratings = read.csv("train_data_100k.csv",
                   colClasses = c("integer","integer","integer","integer"),
                   sep=",",
                   stringsAsFactors = FALSE)

##---------JOin Train Data
user_rating_join_train <- merge(ratings, users_100k, by="user_id")
user_rating_movies_join_train <- merge(user_rating_join_train, movies_100k, by="movie_id")
user_rating_movies_join_train$Zipcode <- NULL
user_rating_movies_join_train$time <- NULL


#user_rating_movies_join_train <- data.frame(sapply(user_rating_movies_join_train,as.factor))
#rules = apriori(user_rating_movies_join_train)
#inspect(rules)


#convert rating-per-row dataframe into sparse User-Item matrix
user_movie_gender_matrix <- as(split(user_rating_movies_join_train[,"movie_id"], user_rating_movies_join_train[,"Occupation"]), "transactions")

#TODO::maxlen can be cahnge between 2 to 6
rule_param = list(
  supp = 0.9,
  conf = 0.8,
  minlen = 2,
  maxlen = 3
)
#TODO::May increase memory or if don't want can comment
memory.limit(size=62000)

#movie_occu <- user_rating_movies_join_train[,c("movie_id","Occupation")]

assoc_rules = apriori(user_movie_gender_matrix,parameter = rule_param)
#TODO:: Need to add belwo apperance attribute and can add different combination.
#assoc_rules = apriori(user_rating_movies_join_train,parameter = rule_param ,
#                      appearance = list(lhs=c("rating=5"), default="rhs"), control = list(verbose=F))


#assoc_rules = apriori(user_rating_movies_join_train,parameter = rule_param ,
#                appearance = list(lhs=c("movie_id=5"), rhs=c("Occupation=educator")), control = list(verbose=F))


summary(assoc_rules)
inspect(assoc_rules)
assoc_rules.sorted <- sort(assoc_rules, by="count")
inspect(assoc_rules.sorted)
write.csv(inspect(assoc_rules),(file="association.csv"))

#read the test data
testegs = read.csv(file="test_data_100k.csv");
##---------JOin Test Data
user_rating_join_test <- merge(testegs, users_100k, by="user_id")
user_rating_movies_join_test <- merge(user_rating_join_test, movies_100k, by="movie_id")

movie_occ_test <- user_rating_join_test[,c("movie_id","Occupation")]


#colnames(testegs) <- c("basketID","items")
colnames(movie_occ_test) <- c("user_id","movie_id")

#execute rules against test data
rulesDF = as(assoc_rules,"data.frame")
testegs$preds = apply(testegs,1,function(X) makepreds(X["movie_id"], rulesDF))

# extract unique predictions for each test user
userpreds = as.data.frame(aggregate(preds ~ user_id, data = testegs, paste, collapse=","))
userpreds$preds = apply(userpreds,1,function(X) uniqueitems(X["preds"]))

# extract unique movie_id bought (or rated highly) for each test user
baskets = as.data.frame(aggregate(movie_id ~ user_id, data = testegs, paste, collapse=","))
baskets$movie_id = apply(baskets,1,function(X) uniqueitems(X["movie_id"]))

#count how many unique predictions made are correct, i.e. have previously been bought (or rated highly) by the user
correctpreds = sum(apply(userpreds,1,function(X) checkpreds(X["preds"],X["user_id"])))

# count total number of unique predictions made
totalpreds = sum(apply(userpreds,1,function(X) countpreds(X["preds"][[1]]))) 
precision = correctpreds*100/totalpreds
cat("precision=", precision, "corr=",correctpreds,"total=",totalpreds)




