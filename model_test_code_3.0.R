# Loading the essential packages
pack_vec <- c("tictoc", "zeallot", "stringr", "quanteda", "tidyr", "dplyr")
sapply(pack_vec, require, character.only = TRUE)
quanteda_options("threads" = 8)
tic("Kneser Ney Time Monitoring")
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
twitter_raw <- get_training_data(twitter_raw, "twitter", sample_percent = 0.4)

# Load profane words list
con <- file("list.txt")
profane_words <- readLines(con)
close(con)

# binding training data together
train_raw <- bind_rows(blogs_raw, news_raw, twitter_raw)
train_raw <- train_raw %>% 
    mutate(mul_groups = as.character(gl(100, k = n()/100)))

library(doParallel)

create_tok_2 <- function(group_number, train_raw, profane_words) {
    corp <- quanteda::corpus(train_raw[train_raw$mul_groups == group_number,])
    corp <- quanteda::corpus_trim(corp, min_ntoken = 10)
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
cl <- makeCluster(8)
registerDoParallel(cl)
train_tok_lst <- parLapply(cl, as.character(seq_len(100)), create_tok_2, train_raw, profane_words)
stopCluster(cl)
toc()
rm(train_raw)
gc()

# Creating Kneser-Ney Smoothing ===================================================================

build_n_gram_model <- function(train_tok, n) {
    n_gram <- quanteda::tokens_ngrams(train_tok, n = n)
    model <- quanteda::dfm(n_gram)
    return(model)
}

tic("Model Training")
cl <- makeCluster(8)
registerDoParallel(cl)
tri_gram_model_shelf <- parLapply(cl, train_tok_lst, build_n_gram_model, n = 3)
bi_gram_model_shelf <- parLapply(cl, train_tok_lst, build_n_gram_model, n = 2)
stopCluster(cl)
toc()

rm(train_tok_lst)
gc()

loop_index <- seq_along(tri_gram_model_shelf)

# Entire Prediction Function Build
kneser_ney_func_1 <- function(index, tri_gram_model_shelf, bi_gram_model_shelf, test_str) {
    
    tri_pattern <- paste0(test_str, "_*")
    bi_pattern <- test_str
    
    tri_gram <- tri_gram_model_shelf[[index]]
    bi_gram <- bi_gram_model_shelf[[index]]
    
    numerator <- Matrix::colSums(quanteda::dfm_select(tri_gram, pattern = tri_pattern)) - 0.75
    
    if (length(numerator) > 0) {
        result <-
            (numerator) / sum(quanteda::dfm_select(bi_gram, pattern = bi_pattern))
        gc()
        return(result)
    } else {
        result <-
            0 / sum(quanteda::dfm_select(bi_gram, pattern = bi_pattern))
        gc()
        return(result)
    }
}

kneser_ney_func_2 <- function(index, tri_gram_model_shelf, bi_gram_model_shelf, test_str) {
    
    tri_pattern <- paste0(test_str, "_*")
    bi_pattern <- test_str
    
    tri_gram <- tri_gram_model_shelf[[index]]
    bi_gram <- bi_gram_model_shelf[[index]]
    
    result <-
        (0.75 / sum(quanteda::dfm_select(bi_gram, pattern = bi_pattern))) * ncol(quanteda::dfm_select(tri_gram, pattern = tri_pattern))
    gc()
    return(result)
    
}

kneser_ney_func_3 <- function(tri_gram, test_str) {
    
    tri_pattern <- paste0(test_str, "_*")
    bi_pattern <- test_str
    
    swap_str <- stringr::str_split(bi_pattern, "_", simplify = TRUE)[1, 1]
    P_name <- colnames(quanteda::dfm_select(tri_gram, pattern = tri_pattern))
    P_lst <- stringr::str_replace(P_name, pattern = swap_str, replacement = "\\.+")
    model_var <- colnames(tri_gram)
    model_var_total <- ncol(tri_gram)
    
    P_continuation_calc <- function(pattern, model_var, model_var_total) {
        numerator <- sum(stringr::str_detect(model_var, pattern = pattern))
        if (numerator > 0) {
            P_continuation <- numerator / model_var_total
            return(P_continuation)
        } else {
            P_continuation <- 0
            return(P_continuation)
        }
    }
    
    cl <- makeCluster(8)
    registerDoParallel(cl)
    third_result <-
        parLapply(cl,
                  P_lst,
                  P_continuation_calc,
                  model_var = model_var,
                  model_var_total = model_var_total)
    stopCluster(cl)
    
    return(third_result)
}

tic("Total Modeling Time")
tic("First and Lambda Part")

cl <- makeCluster(8)
registerDoParallel(cl)
part_1 <- parLapply(cl, loop_index, kneser_ney_func_1, tri_gram_model_shelf, bi_gram_model_shelf, "reduce_your")
lambda_1 <- parLapply(cl, loop_index, kneser_ney_func_2, tri_gram_model_shelf, bi_gram_model_shelf, "reduce_your")
toc()

tic("Third Part")
part_3 <- list()

for (index in loop_index) {
    tri_gram <- tri_gram_model_shelf[[index]]
    part_3[[index]] <- kneser_ney_func_3(tri_gram = tri_gram, test_str = "reduce_your")
}
toc()

for (index in loop_index) {
    item <- unlist(part_3[[index]])
    names(item) <- names(part_1[[index]])
    part_3[[index]] <- item * lambda_1[[index]]
}

part_1_check <- sapply(part_1, length) == 0
part_3_check <- sapply(part_3, length) == 0

total_check <- part_1_check | part_3_check

part_1[total_check] <- NULL
part_3[total_check] <- NULL

part_1_check <- sapply(part_1, function(data) all(complete.cases(data)))
part_3_check <- sapply(part_3, function(data) all(complete.cases(data)))

total_check <- part_1_check == FALSE | part_3_check == FALSE

part_1[total_check] <- NULL
part_3[total_check] <- NULL

part_1_test <- part_1
part_3_test <- part_3


result_prep <- function(part_1, part_3) {
    
    data_reshape <- function(data_lst) {
        data_lst <- lapply(data_lst, function(data) tibble(text = names(data), value = data))
        data_lst <- do.call(bind_rows, data_lst)
        return(data_lst)
    }
    
    part_1 <- data_reshape(part_1)
    part_3 <- data_reshape(part_3)
    
    colnames(part_1) <- c("text_1", "value_1")
    colnames(part_3) <- c("text_3", "value_3")
    
    total_df <- bind_cols(part_1, part_3)
    result <- total_df %>% 
        select(-text_3) %>%
        group_by(text_1) %>% 
        summarise(first_part = mean(value_1), third_part = mean(value_3)) %>% 
        mutate(kneser_ney_smooth = first_part + third_part) %>% 
        arrange(desc(kneser_ney_smooth)) %>% 
        mutate(text_1 = str_replace(text_1, "#", replacement = "")) %>% 
        filter(str_detect(text_1, pattern = "(\\d+|badass.*)", negate = TRUE))
    
    return(result)
}

final_result <- result_prep(part_1_test, part_3_test)


toc()
toc()