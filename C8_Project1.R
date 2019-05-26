# Coursera Data Science Specialization Course 8 Project 1 Script----------------
# Predicting exercise quality in the "Weight Lifting Exercises" dataset


# Acknowledgement for this dataset goes to:
# Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative
# Activity Recognition of Weight Lifting Exercises. Proceedings of 4th
# International Conference in Cooperation with SIGCHI (Augmented Human '13) .
# Stuttgart, Germany: ACM SIGCHI, 2013.
#
# Additional information on their project can be found at the link below:
# http://groupware.les.inf.puc-rio.br/har


# The purpose of this script is to complete the basic requirements behind the
# project 1 peer-graded assignment which is part of the Practical Machine
# Learning course from Johns Hopkins University within the Data Science
# Specialization on Coursera.
#
# The instructions say to build a model that predicts the "classe" variable
# using any of the other variables in the dataset.
#
# The input for this document is the WLE dataset which comes from the URL below.
# The script leaves behind data objects and the model object that best predicted
# the outcome, but the primary purpose is performing the analysis which can
# later be summarized in a markdown file and report.


library(tidyverse)
library(caret)


# Part 0) Function definitions--------------------------------------------------

ConvertDataTypes <- function(data) {
  # Converts data types of the WLE dataset variables, known to introduce NAs
  #
  # Args:
  #   data: a data object to be adjusted
  #
  # Returns:
  #   The data object where the data types of variables are converted
  suppressWarnings(mutate_at(data, c(8:159), as.double)) %>%  # NAs to 33 cols
    mutate_at(c(2, 5:6, 160), as.factor)
}


# Part 1) Loading and preprocessing the data------------------------------------

# Check if the file exists in the directory before downloading it again
training.file <- "pml-training.csv"
if (!file.exists(training.file)) {
  url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(url, training.file)
  rm(url)
}

# Load in the raw training data
raw.train.data <- read.csv(training.file, stringsAsFactors = F)  # 19622 x 160

# Next split off a validation data set from the training data
set.seed(190522)
in.train <- createDataPartition(y = raw.train.data$classe, p = 0.9, list = F)
train.subset <- raw.train.data[in.train, ]  # 17662 obs. of 160 variables
validation.data <- raw.train.data[-in.train, ]  # 1960 obs. of 160 variables
rm(training.file, in.train)


# Part 2) Explore and Process Data----------------------------------------------

# Check out the data
str(train.subset[1:15])
# Some factor variables like user_name and classe were loaded as character type
# and need to be converted. Also, beginning at column 12 some numeric variables
# were loaded as character type and also need conversion. It can be seen that 
# many of these values are simply blank spaces. A function was created to do the
# conversion and easily repeat it later on the validation and test sets. When
# the blank spaces are converted to numeric it introduces NAs by coercion and
# these warnings are suppressed in the function.
train.subset <- ConvertDataTypes(train.subset)  # 17662 obs of 160 variables

# Check the data again
summary(train.subset[1:15])
# It can be seen that some of the variables contain large fractions of NAs now.

# Check for NA values
na.fractions <- train.subset %>%
  map_dbl(function(x) {mean(is.na(x))}) %>%
  subset(. != 0)  # Named num [1:100], the 96 calculated features and 4 var_acc
summary(na.fractions)  # Min. = 0.9796, Max. = 1.0000
# Of the 160 original variables in the data, 100 of them contain NA values, and
# the fraction of NAs in each these columns is over 97%. By reading the research
# paper, one can find that the authors defined "windows" of data (several
# sequential observations) and calculated features for the window. These window
# level summaries represent a second type of data in the dataset.

# Before proceeding it should be determined which type of data should be used as
# the basis for prediction. By checking the test data set it can be seen that
# individual data points are provided (not window summaries). Since the
# prediction model will not be based on window summary data, the variables that
# contain only this data can be excluded.

# Remove window summary columns from the training subset data
na.columns <- names(na.fractions)  # Get the names of the NA columns
train.subset <- select(train.subset, -na.columns)  # 17622 obs. of 60 variables
rm(na.fractions)

# At this point the data are in tidy format, but it is possible that the set
# still contains variables that would cause problems or unnecessary complication
# in the model fitting step. In the section below the variables will be checked
# to see if they are appropriate to keep as potential predictors:
# - The X variable is just an index of the observations, it can be removed.
# - The importance of user_name is explicitly mentioned in the paper, keep it.
# - By checking some plots of the 3 timestamp variables it can be seen that they
# are simply absolute timestamps of the activity. It is conceivable that
# relative timestamps could be useful in model fitting, but the test data do not
# contain enough information to apply that treatment. For this reason these 3
# variables will be removed.
# - new_window simply indicates if that row contains window summary observations
# - Interestingly, it can be shown that the num_window could be a near perfect
# predictor of classe since each window contains just one user and one classe.
# As this is clearly not the intention of the project, this column will be
# removed.
# - The next 52 variables are a series of 13 measurements for each of the 4
# sensor locations. For each sensor location, 4 of the 13 measures are roll,
# pitch, yaw and total acceleration. The remaining 9 are x, y, and z, components
# for each of: gyro, acceleration, and magnet.
# - The last variable classe should be kept as it is the outcome to be predicted

# Based on the checks above, six columns can be removed from the training data:
train.subset <- select(train.subset, c(-1, -(3:7)))

# The data are now tidy and ready for model fitting.


# Part 3) Model Fitting---------------------------------------------------------

# By reading the research paper, one can see that the authors used the random
# forest method when building their own models, so that is clearly a good
# starting point for this step. However, it is well established that the
# boosting method is also a top performing technique for many data sets, so it
# should be considered as well.

# https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80
# The url above is for an article that discusses the strengths and weaknesses of
# the two methods. The article mentions that random forests are well suited to
# multi-class problems like this one. In consideration of this and the selection
# of random forests by the original researchers, that method will be tried here.

# The article also mentions that bias could be introduced by imbalance among the
# factor variables. This data set now contains 2 factor variables, so their
# distribution is checked below.
# plot(train.subset$user_name)  # 6 levels, reasonably even distribution
# plot(train.subset$classe)  # 5 levels, reasonably even distribution
# Based on the check above, model fitting can proceed

# Extract predictors and responses to expedite model fitting
predictors <- select(train.subset, -classe)  # 17662 obs. of 53 variables
response <- train.subset$classe  # Factor with 5 levels, 17662 long
rf.model1 <- train(x = predictors, y = response, method = "rf")
# save(rf.model1, file = "rf.models.RData")
rf.model1$times  #
rf.model1$finalModel  # OOB error rate of
head(getTree(rf.model1$finalModel))


# Need to put subsetting steps into a function






