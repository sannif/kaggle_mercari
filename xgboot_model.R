library(tidyverse)
library(text2vec)
library(xgboost)
library(Matrix)
library(tm)


## Process data
process <- function(fname) {
  df <- read_tsv(paste("../data/", fname, ".tsv", sep=""))
  
  # split category_name into "main category", "category" and "sub category".
  df <- df %>% 
    separate(category_name, into=c("main", "category", "sub_category"), remove=FALSE, sep="/")
  
  df$item_description[df$item_description=="No description yet"] <- ""
  df$brand_name[is.na(df$brand_name)] <- ""
  
  # concatenate name, brand_name and item_description as a unique string
  df <- df %>% mutate(full_text = tolower(paste(name, item_description, brand_name, sep = " ")))
  df$full_text <- sapply(df$full_text, function(i) gsub('[[:digit:]]+', '', i))
  
  df$main[is.na(df$main)] <- "Unknown"
  df$category[is.na(df$category)] <- "Unknown"
  
  df$main <- as.factor(df$main)
  df$category <- as.factor(df$category)
  df$shipping <- as.factor(df$shipping)
  df$item_condition_id <- as.factor(df$item_condition_id)
  
  df
}

## Training
train <- process("train")
# Document-term matrix
stem_tokenizer <- function(x, tokenizer = word_tokenizer) {
  x %>% 
    tokenizer %>% 
    # porter stemmer
      lapply(SnowballC::wordStem, 'en')
}

it_train <-  itoken(train$full_text,
                    tokenizer = stem_tokenizer,
                    ids = train$train_id, 
                    progressbar = TRUE)

stops <- stopwords("en")
vocab <-  create_vocabulary(it_train, stopwords = c(SnowballC::wordStem(stops, 'en'), letters), ngram=(1:2))
pvocab <- prune_vocabulary(vocab, term_count_min = 1000, doc_proportion_max = 1/3)
vectorizer <-  vocab_vectorizer(pvocab)
dtm_train <- create_dtm(it_train, vectorizer)

# TF-IDF
tfidf <- TfIdf$new()
dtm_train <- fit_transform(dtm_train, tfidf)
train <- train %>% select(-c(train_id, category_name, full_text, name, item_description, brand_name, sub_category))


## Test
test <- process("test")

# Document-term matrix
it_test <-  itoken(test$full_text, 
                   tokenizer = word_tokenizer,
                   ids = test$test_id, 
                   progressbar = TRUE)
dtm_test <- create_dtm(it_test, vectorizer)
dtm_test <- fit_transform(dtm_test, tfidf)
test_ids <- test$test_id
test <- test %>% select(-c(test_id, category_name, full_text, name, item_description, brand_name, sub_category))

## Xgboost
log_price <- log(train$price + 1)
train <- sparse.model.matrix(price~.-1, data=train)
train <- cbind(train, dtm_train)
dtrain <- xgb.DMatrix(data=train, label=log_price)
rm(dtm_train) ; gc()

test <- sparse.model.matrix(~.-1, data=test)
test <- cbind(test, dtm_test)
rm(dtm_test); gc()

xgb <- xgb.train(data=dtrain, nrounds=300, verbose=1, subsample=0.8, max_depth=8, eta=c(0.9,0.5), nthread=4, watchlist = list(train=dtrain))

preds <- predict(xgb, test)
preds <- exp(preds) - 1
preds <- ifelse(preds<0, 0, preds)

results <- data.frame(
  test_id = as.integer(test_ids),
  price = preds
)

write.csv(results, file = 'xgboost_model.csv', row.names = FALSE)
