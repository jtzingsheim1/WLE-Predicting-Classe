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
train.data.raw <- read.csv(file.name)
rm(file.name)

# The team that produced the data wrote a paper on their project, and the paper
# contains valuable details on the variables. The variables will be checked more
# closely to understand the references in the paper.
#names.all <- names(train.data)  # 160
#names.belt <- grep("belt", names.all, value = TRUE)  # 38
#names.left1 <- grep("belt", names.all, value = TRUE, invert = TRUE)  # 122
#names.forearm <- grep("forearm", names.left1, value = TRUE)  # 38
#names.left2 <- grep("forearm", names.left1, value = TRUE, invert = TRUE)  # 84
#names.arm <- grep("arm", names.left2, value = TRUE)  # 38
#names.left3 <- grep("arm", names.left2, value = TRUE, invert = TRUE)  # 46
#names.db <- grep("dumbbell", names.left3, value = TRUE)  # 38
#names.left4 <- grep("dumbbell", names.left3, value = TRUE, invert = TRUE)  # 8
# Each sensor location has 38 variables, the eight remaining are not predictors
# What are the 38 variables for each sensor?
# Euler angles: roll, pitch, and yaw
#test <- grep("avg", names.belt, value = TRUE, invert = TRUE) %>%
#  # Variance: roll, pitch, and yaw
#  grep("stddev", ., value = TRUE, invert = TRUE) %>%
#  grep("max", ., value = TRUE, invert = TRUE) %>%
#  grep("min", ., value = TRUE, invert = TRUE) %>%
#  grep("amplitude", ., value = TRUE, invert = TRUE) %>%
#  grep("kurtosis", ., value = TRUE, invert = TRUE) %>%
#  grep("skewness", ., value = TRUE, invert = TRUE) %>%
#  # There are roll, pitch, and yaw variables for 9 different feature types, 27
#  # Accelerometer: x, y, and z
#  grep("gyros", ., value = TRUE, invert = TRUE) %>%
#  grep("magnet", ., value = TRUE, invert = TRUE)
#  # There are also x, y, and z variables for three feature types, 9
#  # This accounts for 36 of the 38 variables, and the remaining 2 are:
#  # total_accel_* and var_total_accel_*
#rm(test)

# The 8 non-predictors are:
# - X: The index of the observation, ranges from 1 to 19622
# - user_name: The name of one of the six males performing the test
# - timestamp_part_1
# - timestamp_part_2
# - cvtd_timestamp
# - new_window
# - num_window
# - classe

# Check out NA values
#train.NA.cols <- train.data.raw %>%
#  map_dbl(NAFraction) %>%
#  enframe() %>%
#  rename(variable = name, na.fraction = value) %>%
#  filter(na.fraction != 0)

#train.complete.cols <- train.data.raw %>%
#  map_dbl(NAFraction) %>%
#  enframe() %>%
#  rename(variable = name, na.fraction = value) %>%
#  filter(na.fraction == 0)

# There are two types of data in this dataset, and they become noticeable by
# checking the pattern of the NA values. 67 of the variables contain 97.9% NA
# values, and the exact number of NAs is the same for each variable. Meanwhile
# the remaining 93 variables all contain 0 NAs. By checking the supporting
# material from the original researchers, one can see that the variables with
# large NA fractions are variables summarizing a "window" of data.

# To see which type of data are needed for this assignment, the test data can be
# checked. The test set shows that the predictions will not be based on the
# summary data, so for this reason those columns can be excluded.

# Get column names for the summary variables
NA.cols <- train.data.raw %>%
  map_dbl(NAFraction) %>%
  subset(. != 0) %>%
  names()

# Subset the training data
train.data <- train.data.raw %>%
  select(-NA.cols) %>%
  as_tibble()






# Check how variables change with time in a given window:
#test3 <- filter(train.data.raw, num_window == 1)
#plot(test3$roll_belt ~ test3$raw_timestamp_part_2)
#plot(test3$pitch_belt ~ test3$raw_timestamp_part_2)
#plot(test3$yaw_belt ~ test3$raw_timestamp_part_2)
#plot(test3$roll_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$pitch_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$yaw_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$kurtosis_roll_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$kurtosis_picth_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$kurtosis_yaw_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$skewness_roll_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$skewness_pitch_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$skewness_yaw_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$max_yaw_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$min_yaw_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$amplitude_yaw_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$total_accel_dumbbell ~ test3$raw_timestamp_part_2)
#plot(test3$gyros_dumbbell_x ~ test3$raw_timestamp_part_2)
#plot(test3$gyros_dumbbell_y ~ test3$raw_timestamp_part_2)
#plot(test3$gyros_dumbbell_z ~ test3$raw_timestamp_part_2)
#plot(test3$accel_dumbbell_x ~ test3$raw_timestamp_part_2)
#plot(test3$accel_dumbbell_y ~ test3$raw_timestamp_part_2)
#plot(test3$accel_dumbbell_z ~ test3$raw_timestamp_part_2)
#plot(test3$magnet_dumbbell_x, test3$raw_timestamp_part_2)
#plot(test3$magnet_dumbbell_y, test3$raw_timestamp_part_2)
#plot(test3$magnet_dumbbell_z, test3$raw_timestamp_part_2)
#plot(test3$magnet_dumbbell_z, test3$raw_timestamp_part_2)



















