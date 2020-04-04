# Loading the essential packages
pack_vec <- c("tictoc", "zeallot", "stringr", "tidyr", "dplyr", "NextWord")
sapply(pack_vec, require, character.only = TRUE)
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
blogs_raw <- get_training_data(blogs_raw, "blogs", sample_percent = 0.8)
news_raw <- get_training_data(news_raw, "news", sample_percent = 0.8)
twitter_raw <- get_training_data(twitter_raw, "twitter", sample_percent = 0.8)

total_raw <- bind_rows(blogs_raw, news_raw, twitter_raw) %>% 
    group_by(text_type) %>% 
    sample_n(50000) %>% 
    mutate(text_tag = gl(10, k = n() / 10)) %>% 
    ungroup()

test_text_lst <- lapply(1:10, function(i) total_raw[total_raw$text_tag == i, ]$text)
rm(blogs_raw, news_raw, twitter_raw, total_raw)
gc()

library(doParallel)

con <- file("list.txt")
profane_words <- readLines(con)
close(con)

tic("clean text")
cl <- makeCluster(7)
registerDoParallel(cl)
test_text <- parLapply(cl, test_text_lst, cleanText, wordsToRemove = profane_words)
stopCluster(cl)
toc()

tic("tag text")
cl <- makeCluster(7)
registerDoParallel(cl)
test_text <- parSapply(cl, test_text_lst, tagText)
stopCluster(cl)
toc()

join_prob_calc <- function(n_gram, clean_text_lst) {
    ngram_frame <- NextWord::ngram(clean_text_lst, n_gram)
    t_n_tok_pro <-NextWord::calcProbsWithPos(ngram_frame)
    t_n_pos_pro <- NextWord::getPosProbs(ngram_frame)
    t_n_join_pro <- NextWord::joinProbs(t_n_tok_pro, t_n_pos_pro)
    return(t_n_join_pro)
}

tic("Get Joint Probability List")
cl <- makeCluster(7)
registerDoParallel(cl)
join_prob_lst <- parLapply(cl, 1:4, join_prob_calc, clean_text_lst = test_text)
stopCluster(cl)
toc()

tic("Model Building")
test_model <- buildModel(join_prob_lst)
toc()

saveRDS(test_model, file = "git_model_beta1_L.rds")

predict(test_model, "I am really", cleanedPosText = FALSE)
