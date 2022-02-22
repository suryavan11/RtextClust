############ global parameters #################
use_POS = T  ## F,F, T, T
remove_stopwords = F ## T,F,T,F
max.ngrams = 2
wv.dim = 300
call_level = TRUE
# remove.duplicate.2grams = T ## removes netflix_netflix etc

############## Load Libraries ##################
# devtools::install_github("mukul13/rword2vec")
library(rword2vec)
library(text2vec)
library(stopwords)
library(tokenizers)
library(cld2)
library(future)
library(future.apply)
library(keras)
use_condaenv("r-tensorflow")
library(spacyr)
## in anaconda prompt create a backup of r-reticulate : conda create --name r-reticulate-bkup --clone r-reticulate
#### then run R as administrator. this is done only once:  spacy_install(python_version = '3.6.10', envname = 'r-reticulate') 
library(tictoc)
library(tidyverse)

seed = 80
options(scipen=999) 
options(future.globals.maxSize = +Inf) 
setwd('...')
source('utils.R')

#### read and preprocess data ########################
stopwords_longlist = stopwords('en')
df = read_delim('...', delim = '|')
df$endoffset = 1 
# df = df[1:1000,]

# df_bkup = df

colnames(df) = tolower(colnames(df))

if(use_POS == T & remove_stopwords == T) {
stopwords_longlist = attach_POS2(stopwords_longlist, python.env = 'r-tensorflow', chunksize = 50000)
}

process_upto_POS = function(dat, part.of.speech = T,
                            remove_stopwords = F,
                            stopword.list = stopwords('en'),
                            input.file = NULL, output.file = NULL){
  #browser()
  dat = dat%>%
    mutate(orig.phrase = phrase)%>%
    mutate(phrase = tolower(iconv(phrase, to = "UTF-8")) )%>% ## cleans up unwanted characters
    mutate(phrase = str_trim(str_replace_all(phrase, '\\s+', ' ')))
  
  if (call_level == T) {
    dat = dat%>%
    group_by(sourcemediaid)%>%
    arrange(endoffset, .by_group = TRUE)%>%
     summarize(phrase = str_c( phrase, collapse = ' ')  )%>%
    ungroup()
  } 
    
    dat = dat%>%
    mutate(phrase = preprocess.text.fn(phrase))%>%
    mutate(cleaned.phrase = phrase)%>%
    group_by(sourcemediaid)%>%
    mutate(language = cld2::detect_language(str_c( phrase, collapse = ' '), lang_code = FALSE))%>%
    ungroup() %>%
    filter(str_count(phrase, '\\w+') > 1 & language == 'ENGLISH')

  dat = dat%>%
    select(phrase)%>%
    separate_rows(phrase, sep = '\\.')%>%
    mutate(phrase = str_trim(phrase))%>%
    filter(phrase != "")
  
  if (part.of.speech == T) {
  ### watch for RAM usage for the function below. If RAM is maxing out and disk usage goes up, then the function will slow down drastically
  dat$phrase = suppressMessages(suppressWarnings(
    attach_POS2(dat$phrase, python.env = 'r-tensorflow', chunksize = 50000 )  ))
  } 
  
  
  if(remove_stopwords == T) {
    if(part.of.speech == T) {
      tokens = tokenize_regex(dat$phrase ) 
    } else {
      tokens = tokenize_word_stems(dat$phrase ) 
    }
    # tokens = lapply(tokens,function(x) x[nchar(x)>2])
    # tokens = lapply(tokens,function(x) x[nchar(x)<= 20])
    tokens = lapply(tokens,function(x) x[!x %in% stopword.list])
    dat$phrase = sapply(tokens, function(x) paste0(x, sep=' ',collapse=' '))
  }
 
  
  dat = dat%>%filter(phrase != "")
  
  print(paste('chunk ', ctr, " has been processed"))
  ctr <<- ctr+1
  
  return(dat)
}

tic()
chunk_size = 1e6 ## this number along with the chunk size of the attach_POS2 function in 'process_upto_POS' controls the RAM usage. Adjust the chunk size in the attach_POS2 function first before this one
ctr = 1
chunk_arr = (1:nrow(df))%/%( nrow(df) / (ceiling(nrow(df)/chunk_size)) + 0.01)
df = lapply(split(df, chunk_arr ) , process_upto_POS, use_POS, 
            remove_stopwords, stopwords_longlist,NULL, NULL )
df = data.table::rbindlist(df, use.names = T)
toc()


### write the training data file
write_lines(df$phrase, 'data/training_sentences.txt', sep="\n" )

rm(df)

 
############## word2vec
if(max.ngrams == 2) {
  word2phrase(train_file = 'data/training_sentences.txt',
              output_file = 'data/training_sentences_w2v_phrase.txt')
  filepath = 'data/training_sentences_w2v_phrase.txt'
} else {
  filepath = 'data/training_sentences.txt'
}

model=word2vec(train_file = filepath,
               output_file = "models/w2v_phrase.bin", ## w2v_2020-01-07.bin, model/w2v_2020-01-14.bin
               window = 5,
               layer1_size = wv.dim,
               min_count=50,
               binary=1, num_threads=10)


# dist=distance(file_name = "models/w2v_phrase.bin",search_word = "atheist",num = 30)
# dist

###convert .bin to .txt
bin_to_txt("models/w2v_phrase.bin","models/w2v_phrase.txt")


#### save as RDS in a format that can be used by earlier code (words are rownames and column names start from V1 to V...) )
data1=read_delim("models/w2v_phrase.txt",skip=1,
                 col_names = FALSE,
                 delim=' ', trim_ws = FALSE)%>%
  mutate(X1 = str_replace_all(X1,'(.*?_[A-Z]+)_(.*)','\\1~\\2'))%>%
  na.omit()%>% ### the last row is incomplete for some reason. na.omit will remove it
  column_to_rownames('X1')%>%
  `colnames<-`(paste0('V',1:ncol(.)) )

saveRDS(as.matrix(data1), paste0('models/w2v_',
                             ifelse(max.ngrams==2,'phrase_',''),
                             ifelse(remove_stopwords==T,'stop_','nostop_'),
                             ifelse(use_POS==T,'pos_','nopos_'),
                             wv.dim,'dim.RDS'))
