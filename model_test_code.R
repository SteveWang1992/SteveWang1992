# Loading the essential packages
pack_vec <- c("tictoc", "zeallot", "stringr", "quanteda", "tidyr", "dplyr", "ggplot2")
sapply(pack_vec, require, character.only = TRUE)
quanteda_options("threads" = 8)

text_files <- list.files("final/en_US")
prefix <- paste0(getwd(), "/final/en_US/")
text_files <- paste0(prefix, text_files)

# Building the text loading function
raw_text_load <- function(file_path) {
    con <- file(file_path)
    raw_text <- readLines(con, skipNul = TRUE)
    close(con)
    return(raw_text)
}

# Loading all the text files into vector
tic("Raw Text Loading")
c(blogs_raw, news_raw, twitter_raw) %<-% lapply(text_files, raw_text_load)
toc()

# Sample The Raw Text Data
set.seed(2020)
get_training_data <- function(raw_data, text_type, sample_percent = 0.1) {
    sample_index <- as.logical(rbinom(length(raw_data), size = 1, prob = sample_percent))
    text <- raw_data[sample_index]
    df <- tibble(text = text, text_type = rep(text_type, length(text)))
}

# Get training data for each text type
blogs_raw <- get_training_data(blogs_raw, "blogs", sample_percent = 0.2)
news_raw <- get_training_data(news_raw, "news", sample_percent = 0.2)
twitter_raw <- get_training_data(twitter_raw, "twitter", sample_percent = 0.2)

# Load profane words list
con <- file("list.txt")
profane_words <- readLines(con)
close(con)

# binding training data together
train_raw <- bind_rows(blogs_raw, news_raw, twitter_raw)
train_raw <- train_raw %>% 
    mutate(mul_groups = as.character(gl(10, k = n()/10)))

# Tokens group building
# create_tok function
library(doParallel)

create_tok <- function(group_number) {
    corpus(train_raw[train_raw$mul_groups == group_number, ]) %>%
        corpus_trim(min_ntoken = 2) %>%
        tokens(
            remove_punct = TRUE,
            remove_symbols = TRUE,
            remove_numbers = TRUE
        ) %>%
        tokens_remove(pattern = profane_words)
}

create_tok_2 <- function(group_number, train_raw, profane_words) {
    corp <- quanteda::corpus(train_raw[train_raw$mul_groups == group_number,])
    corp <- quanteda::corpus_trim(corp, min_ntoken = 2)
    tok <- quanteda::tokens(
        corp,
        remove_punct = TRUE,
        remove_symbols = TRUE,
        remove_numbers = TRUE
    )
    tok <- quanteda::tokens_remove(tok, pattern = profane_words)
    return(tok)
}

tic("Tokenization")
cl <- makeCluster(7)
train_tok_lst <- parLapply(cl, as.character(seq_len(10)), create_tok_2, train_raw, profane_words)
stopCluster(cl)
toc()
rm(train_raw)
gc()

# Create the token
tic("Tokens Building")
train_tok <- corpus(train_raw[train_raw$mul_groups == 1]) %>%
    corpus_trim(min_ntoken = 2) %>%
    tokens(
        remove_punct = TRUE,
        remove_symbols = TRUE,
        remove_numbers = TRUE
    ) %>% tokens_remove(pattern = profane_words)
toc()
gc()

# Building the stupid backoff model ========================================================================
# Parallel Model

tri_gram_model <- function(tok) {
    tok_ngram <- quanteda::tokens_ngrams(tok, n = 3)
    model <- quanteda::dfm(tok_ngram)
    return(model)
}

# Parallel computiing
tic("Modeling Training")
cl <- makeCluster(7)
model_lst <- parLapply(cl, train_tok_lst, tri_gram_model)
stopCluster(cl)
toc()

model_pred <- function(model) {
    result <- quanteda::dfm_select(model, pattern = "like_Chicken_*")
    result <- base::colSums(result)
    result <- result[base::order(result, decreasing = TRUE)]
    return(result)
}

tic("Predication")
cl <- makeCluster(7)
registerDoParallel(cl)
pred_result <- lapply(model_lst, model_pred)
stopCluster(cl)
toc()

# Building the 3gram model ========================================================================
library(doParallel)
ncores <- 7
cl <- makeCluster(ncores)
registerDoParallel(cl)
tic("Model Building")
four_gram_model <- train_tok %>% tokens_ngrams(n = 3) %>% dfm()
toc()
tic("Predication")
test <- four_gram_model %>% dfm_select(pattern = "like_Chicken_*") %>% colSums()
test[order(test, decreasing = TRUE)]
toc()
stopCluster(cl)

