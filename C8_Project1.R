# Coursera Data Science Specialization Course 8 Project 1 Script----------------
# Predicting exercise quality with in the "Weight Lifting Exercises" dataset


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
# The script leaves behind the data object and the model object that best
# predicted the outcome, but the primary purpose is performing the analysis
# which can later be summarized in a markdown file and report.


library(tidyverse)


# Part 0) Function definitions--------------------------------------------------

NAFraction <- function(x) {
  # Calculates the NA fraction of an object
  #
  # Args:
  #   x: an R object such as a vector
  #
  # Returns:
  #   A numeric value indicating the fraction of NA values in the object
  mean(is.na(x))
}

# Part 1) Loading and preprocessing the data------------------------------------

file.name <- "pml-training.csv"
# Check if the file exists in the directory before downloading it again
if (!file.exists(file.name)) {
  url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(url, file.name)
  rm(url)
}

# Load in the training data
train.data <- read.csv(file.name, stringsAsFactors = FALSE) %>%  # 19622 x 160
  mutate_at(c(2, 5:6, 160), as.factor) %>%
  mutate_at(c(8:159), as.double)  # 33 columns had NAs introduced by coercion
rm(file.name)

# Check for NA values
NA.cols <- train.data %>%
  map_dbl(NAFraction) %>%
  subset(. != 0) %>%
  names()  # 100 variables, the 96 and the var_accel for each sensor

# There are two types of data in this dataset, and the difference can be
# observed by checking the pattern of NA values. Several variables contain 98%
# or more NAs. The research paper explains that these variables summarize a
# "window" of data.

# To see which type of data are needed for this assignment the test data can be
# checked. It can be seen that the test set do not contain summary data, so the
# prediction model should not be built with the summary variables, and for this
# reason they can be excluded.

# Subset the training data
train.data <- train.data %>%
  select(-NA.cols) %>%
  as_tibble()  # 19622 obs. of 60 variables
# At this point the data are tidy
rm(NA.cols, NAFraction)


# Part 2) Additional Data Processing--------------------------------------------

# Which variables should be eliminated before splitting the data and fitting a
# model?
#summary(train.data$X)  # X is just an index of the observations, eliminate it
# The importance of user_name is explicitly mentioned in the paper, keep it

# The raw_timestamp variables are related to the abosolute time that the act was
# performed, but they would contribute little to a prediction model as is
#plot(train.data$raw_timestamp_part_1 ~ train.data$X)  # 4 horizontal dash lines
#plot(train.data$raw_timestamp_part_2 ~ train.data$X)  # Dense noise
#plot(train.data$raw_timestamp_part_2 ~ train.data$raw_timestamp_part_1)  # 6v ln
#filter(train.data, user_name == "adelmo") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # Noise
#filter(train.data, user_name == "carlitos") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # Noise
#filter(train.data, user_name == "charles") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # Noise
#filter(train.data, user_name == "eurico") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # Noise
#filter(train.data, user_name == "jeremy") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # Noise
#filter(train.data, user_name == "pedro") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # Noise
#filter(train.data, user_name == "adelmo" | user_name == "carlitos") %>%
#  plot(raw_timestamp_part_2 ~ raw_timestamp_part_1, data = .)  # 2 v lines

# Check cvtd_timestamp
#train.data %>%
#  group_by(cvtd_timestamp, user_name) %>%
#  summarize(count = n()) %>%
#  spread(key = user_name, value = count)
# This is just the absolute timestamp of the exercise, not important for model

# new_window simply indicates if that row contains summary observations or not

# How many users are in each window?
#train.data %>%
#  group_by(num_window, user_name) %>%
#  summarize() %>%
#  group_by(num_window) %>%
#  summarize(unique.users = n()) %>%
#  select(unique.users) %>%
#  max()  # 1, each window only has one user
# How many exercise types are in each window?
#train.data %>%
#  group_by(num_window, classe) %>%
#  summarize() %>%
#  group_by(num_window) %>%
#  summarize(unique.exer = n()) %>%
#  select(unique.exer) %>%
#  max()  # 1, each window only has one exercise
# If each window is one user doing one exercise it ends up as a perfect
# predictor of the classe variable. This of course is not the intention of the
# assignment, so this variable will be removed.

# The next 52 variables are measurements, check what they are 
#td.names <- names(train.data)  
#length(grep("belt", td.names))  # 13
#length(grep("forearm", td.names))  # 13
#length(grep("_arm", td.names))  # 13
#length(grep("dumbbell", td.names))  # 13
#rm(td.names)
# The next 52 variables are a series of 13 measurements for each of the 4 sensor
# locations. For each sensor location, 4 of the 13 measures are roll, pitch, yaw
# and total acceleration. The remaining 9 are x, y, and z, components for each
# of: gyro, acceleration, and magnet.

# The final variable, classe, is of course the outcome to be predicted.















