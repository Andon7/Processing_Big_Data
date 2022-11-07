library(tidyverse)
library(tidyr)
library(caret)
library(caTools)
library(stats)
library(e1071)

hotel_booking <- read_csv(here::here("hotel_booking.csv"))


#Missing values

sapply(hotel_booking, function(x) sum(is.na(x)))

#The variable company has 112.593 missing, we need to drop it.

#Also, as we  can see that some variables are not useful in predicting and need to be dropped.

hotel <- hotel_booking %>%
  select(-name, -email, -"phone-number", -credit_card, -company, -agent) 

#Fill missing values for the variables country and children 

hotel <- hotel %>% 
  mutate(country = replace_na(country, "none")) %>%
  mutate(children = replace_na(children, 0)) 

sapply(hotel, function(x) sum(is.na(x)))



# Logistic regression ( predict is_cancelled) -------------------------------------

# We are predicting whether the reservation was cancelled

# We do not include reservation status as it obviously related with the variable we want to predict

first <- hotel %>%
  select(hotel, is_canceled, lead_time, arrival_date_year, arrival_date_month, 
         arrival_date_week_number,stays_in_weekend_nights,  stays_in_week_nights,
         adults, is_repeated_guest,previous_cancellations, 
         previous_bookings_not_canceled, booking_changes, deposit_type, customer_type,
         adr)



#One-hot encoding
dummy_first <- dummyVars( ~ ., data=first)

first <- data.frame(predict(dummy_first, newdata=first)) %>%
  select(-hotelResort.Hotel) %>%
  mutate(is_canceled = as.factor(is_canceled))

#Split the data 80% - 20%
set.seed(123)
trainIndex <- createDataPartition(first$is_canceled, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- first[ trainIndex,]
test  <- first[-trainIndex,]


#Logistic regression (predict is_canceled)
#Train the model using the training data
logistic_model <- glm(
  is_canceled ~ ., data=train, family="binomial")
summary(logistic_model)

#Run the test data through the model
predictions <- predict(logistic_model, test, type="response")

confusion_matrix <- table(Actual_value = test$is_canceled, predicted_value = predictions > 0.5)

#Accuracy 
accuracy <- (confusion_matrix[[1,1]] + confusion_matrix[[2,2]]) / sum(confusion_matrix)
# 0.7740503


#Linear regression (predict adr) ----------------------------------
#How much do they pay per day?

second <- hotel %>%
  #removing weird values
  filter(adr < 500,
         adr > 0) %>%
  mutate(total_nights = stays_in_weekend_nights + stays_in_week_nights) %>%
  select(hotel, arrival_date_month, stays_in_weekend_nights,
         total_nights, adults, meal, reserved_room_type, is_canceled, 
         adr)

#One-hot encoding
dummy_second <- dummyVars( ~ ., data=second)

second <- data.frame(predict(dummy_second, newdata=second)) %>%
  select(-hotelResort.Hotel) 

#Split the data 80% -20%
set.seed(123)
trainIndex <- createDataPartition(second$adr, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train <- second[ trainIndex,]
test  <- second[-trainIndex,]


#Linear regression (predict adr)
#Train the model using the training data
linear_model <- lm(
  adr ~ ., data=train)

summary(linear_model)

predictions <- predict(linear_model, test)
rmse <- sqrt(mean((test$adr -predictions)^2))
# 33.46164


# SVM (predict is_repeated_guest) --------------------------------

# We want to predict if a guest has come again to the hotel.
# We do not include the variables previous_cancellations and previous_bookings_not_canceled 
# As they obviously related with the variable we want to predict

third <- hotel %>%
  mutate(total_nights = stays_in_weekend_nights + stays_in_week_nights) %>%
  select(total_nights, is_canceled, arrival_date_year, arrival_date_month, 
         arrival_date_week_number, arrival_date_day_of_month, 
         meal, is_repeated_guest, reserved_room_type, deposit_type)

#One-hot encoding
dummy_third <- dummyVars( ~ ., data=third)         

third <- data.frame(predict(dummy_third, newdata=third))

third$is_repeated_guest <- as.factor(third$is_repeated_guest)

#Split the data 60% - 40%
set.seed(123)
trainIndex <- createDataPartition(third$is_repeated_guest, p = .6, 
                                  list = FALSE, 
                                  times = 1)
train <- third[ trainIndex,]
test  <- third[-trainIndex,]

# Due to the imbalance on the variable we want to predict 
# we need to weight accordingly the two classes

svm_model <- svm(is_repeated_guest ~., data=train, kernel = "linear",
                 cost = .001, scale = FALSE, class.weights = c("0"=1, "1"=30))
print(svm_model)

# optimize for the cost parameter
tuned <- tune(svm, is_repeated_guest ~., data=train, kernel = "linear",
              ranges=list(cost = c(0.001, 0.01, 1, 10, 100)))

summary(tuned)
# best cost = 0.001

prediction <- predict(svm_model, test, type="Class")

final <- data.frame(actual = test$is_repeated_guest, predicted = prediction)

confusion_matrix <- table(final)
confusion_matrix
#Accuracy 
accuracy <- (confusion_matrix[[1,1]] + confusion_matrix[[2,2]]) / sum(confusion_matrix)
# 0.6691794

# Due to the imbalance data set accuracy is a misleading metric. A dummy model predicting always the 
# negative class would achieve an accuracy of 97%. So we use precision and recall instead

precision <- confusion_matrix[[2,2]] / (confusion_matrix[[2,2]] + confusion_matrix[[1,2]])
recall <- confusion_matrix[[2,2]] / (confusion_matrix[[2,2]] + confusion_matrix[[2,1]])



#Clustering (k-means) [adr &lead_time ] ------------------------------

fourth <- hotel %>%
  select(adr, lead_time) %>%
  filter(adr < 500,
         adr > 0)

plot(fourth$adr, fourth$lead_time)

# Decide on the best value for k
total <- list()
for(k in 1:10){
  k_means <- kmeans(fourth, centers = k)
  sum_squares <- sapply(k_means[4], sum)
  total <- append(total, sum_squares)
}

plot(c(1:10), total)

# Using the elbow method we find that the best value for k is k=3
final_clustering <- kmeans(fourth, centers = 3)


fourth <- cbind(fourth, cluster = final_clustering$cluster)

ggplot(fourth, aes(x=adr, y=lead_time, color=cluster)) +geom_point() 
