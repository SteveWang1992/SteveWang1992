# Loading the essential packages
pack_vec <- c("tictoc", "zeallot", "stringr", "quanteda", "tidyr", "dplyr", "ggplot2")
sapply(pack_vec, require, character.only = TRUE)
quanteda_options("threads" = 8)
tic("Total")
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
twitter_raw <- get_training_data(twitter_raw, "twitter", sample_percent = 0.5)

# Load profane words list
con <- file("list.txt")
profane_words <- readLines(con)
close(con)

# binding training data together
train_raw <- bind_rows(blogs_raw, news_raw, twitter_raw)
train_raw <- train_raw %>% 
    mutate(mul_groups = as.character(gl(10, k = n()/10)))

rm(blogs_raw, news_raw, twitter_raw)
gc()

# Create tokenization from raw data and get rid of raw data ========================================================================

library(doParallel)

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

# Building the stupid backoff model ========================================================================
# Parallel Model

# Building Trigram, Bigram, Unigram Model
model_shelf <- function(tok) {
    trigram_model <- quanteda::dfm(quanteda::tokens_ngrams(tok, n = 3))
    bigram_model <- quanteda::dfm(quanteda::tokens_ngrams(tok, n = 2))
    unigram_model <- quanteda::dfm(quanteda::tokens_ngrams(tok, n = 1))
    model_shelf <- list(trigram_model = trigram_model, 
                        bigram_model = bigram_model, 
                        unigram_model = unigram_model)
    return(model_shelf)
}

# Model Training
tic("Modeling Training")
cl <- makeCluster(8)
model_lst <- parLapply(cl, train_tok_lst, model_shelf)
stopCluster(cl)
toc()

# Building Predict Function
model_pred <- function(model_shelf, tri_pattern = NULL, bi_pattern = NULL, func = "trigram") {
    
    suffix <- "_*"
    tri_pattern <- paste0(tri_pattern, suffix)
    bi_pattern <- paste0(bi_pattern, suffix)
    
    trigram_pred <- function(model_shelf) {
        result <- quanteda::dfm_select(model_shelf$trigram_model, pattern = tri_pattern)
        result <- Matrix::colSums(result)
        result <- result[base::order(result, decreasing = TRUE)]
        return(result)
    }
    
    bigram_pred <- function(model_shelf) {
        result <- quanteda::dfm_select(model_shelf$bigram_model, pattern = bi_pattern)
        result <- Matrix::colSums(result)
        result <- result[base::order(result, decreasing = TRUE)]
        return(result)
    }
    
    unigram_pred <- function(model_shelf) {
        result <- Matrix::colSums(model_shelf$unigram_model)
        result <- result[base::order(result, decreasing = TRUE)][1:10]
        return(result)
    }
    # Switch to different function based on func argument
    if (func == "trigram") {
        prediction <- trigram_pred(model_shelf)
        return(prediction)
    } else if (func == "bigram") {
        prediction <- bigram_pred(model_shelf)
    } else {
        prediction <- unigram_pred(model_shelf)
    }
}

# Putting predict function into parallel computing =================================================================

parallel_predict <- function(cl, model_lst, model_pred, ..., backoff = FALSE) {
    
    backoff_running <- function(cl,
                                model_lst,
                                model_pred, ..., func = "trigram") {
        pred_result <-
            parLapply(cl, model_lst, model_pred, ..., func)
        return(pred_result)
    }
    
    pred_result_check <- function(pred_result) {
        check_length <- sapply(pred_result, length)
        check_length <- all(check_length == 0)
        return(check_length)
    }
    
    backoff_result <- function(pred_result, backoff) {
        result_combine <- do.call(bind_rows, pred_result)
        result_combine[is.na(result_combine)] <- 0
        
        result_combine <- result_combine %>%
            summarise_all(sum) %>%
            gather(key = "var", value = "number") %>%
            mutate(prob = number / sum(number)) %>%
            top_n(10, number)
        
        if (backoff) {
            result_combine <- result_combine %>% 
                mutate(prob = 0.4 * prob)
            return(result_combine)
        } else {
            return(result_combine)
        }
    }
    
    pred_result <- backoff_running(cl, model_lst, model_pred, ...)
    
    if (pred_result_check(pred_result)) {
        # Back off to bi_gram model
        pred_result <- backoff_running(cl, model_lst, model_pred, ..., func = "bigram")
        
        if (pred_result_check(pred_result)) {
            # Back off to uni_gram model
            pred_result <- backoff_running(cl, model_lst, model_pred, ..., func = "unigram")
            # Return Unigram Result
            final_result <- backoff_result(pred_result, backoff = TRUE)
            return(final_result)
        } else {
            # Return Bigram Result
            final_result <- backoff_result(pred_result, backoff = TRUE)
            return(final_result)
        }
    } else {
        # Return Trigram Result
        final_result <- backoff_result(pred_result, backoff)
        return(final_result)
    }
}

tic("Stupid Back Model Prediction")
cl <- makeCluster(8)
registerDoParallel(cl)
prediction_result <- parallel_predict(cl, model_lst, model_pred, tri_pattern = "settle_the", bi_pattern = "the")
stopCluster(cl)
toc()
toc()