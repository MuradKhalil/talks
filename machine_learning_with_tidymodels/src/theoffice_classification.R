library(conflicted)
library(tidyverse)
library(tidymodels)
library(textrecipes)
library(janitor)

df <- schrute::theoffice
glimpse(df)

characters <- c("Pam", "Dwight")

mydf <- df %>%
  dplyr::filter(character %in% characters) %>%
  select(text, character) %>%
  mutate(character = as_factor(if_else(character == "Pam", 1, 0)))

train_test_splits <- initial_split(mydf, prop = 0.75, strata = character)
train_test_splits

training_data <- training(train_test_splits)
nrow(training_data)

testing_data <- testing(train_test_splits)
nrow(testing_data)

blueprint <- recipe(character ~ text, data = training_data) %>%
  step_tokenize(text) %>%
  step_lemma(text) %>%
  step_stopwords(text) %>%
  step_tokenfilter(text, max_tokens = 500) %>%
  step_tfidf(text)

prepped <- prep(blueprint, training = training_data, strings_as_factors = T)
prepped

juiced_train <- juice(prepped)
juiced_train

baked_test <- bake(prepped, new_data = testing_data)
baked_test

glm_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

validation_splits <- vfold_cv(juiced_train, v = 5, strata = character)
validation_splits

cv_results <- fit_resamples(
    glm_spec,
    character ~ ., 
    validation_splits,
    control = control_resamples(save_pred = T)
    )

cv_results %>%
  collect_predictions() %>%
  conf_mat(character, .pred_class)

cv_results %>%
  collect_metrics()

glm_fit <- glm_spec %>%
  fit(character ~ ., data = juiced_train)

glm_fit

glm_fit %>%
  predict(new_data = baked_test, type = "prob") %>%
  mutate(true_label = baked_test$character) %>%
  roc_auc(true_label, .pred_1)