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

blogs_raw <- get_training_data(blogs_raw, "blogs", sample_percent = 0.1)
news_raw <- get_training_data(news_raw, "news", sample_percent = 0.08)
twitter_raw <- get_training_data(twitter_raw, "twitter", sample_percent = 0.3)

# Combined dataframes
train_data <- bind_rows(blogs_raw, news_raw, twitter_raw)
rm(blogs_raw, news_raw, twitter_raw)
gc()

# Build Corpus
tic("Corpus Building")
train_corp <- corpus(train_data)
toc()

# Token Model Prep
tic("Tokenization")
train_token <- train_corp %>% 
    corpus_reshape(to = "sentences") %>% 
    corpus_trim(what = "sentences", min_ntoken = 2) %>% 
    tokens(remove_numbers = TRUE, remove_punct = TRUE)
toc()
rm(train_corp)
gc()

# Trigram
tic("Trigram")
test <- train_token %>% 
    tokens_ngrams(n = 3) %>% 
    dfm() %>% 
    dfm_select(pattern = "quite_some_*") %>% 
    colSums()
test[order(test, decreasing = TRUE)]
toc()

gc()
# Four gram
tic("Five Gram")
test <- train_token %>% 
    tokens_ngrams(n = 5) %>% 
    dfm() %>% 
    dfm_select(pattern = "romantic_date_at_the_*") %>% 
    colSums()
test[order(test, decreasing = TRUE)]
toc()

