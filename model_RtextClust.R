############ global parameters #################
rm(list = ls(all = TRUE))
gc()
use_POS = T  
remove_stopwords = F 
max.ngrams = 2
wv.dim = 300

## This is to turn off scientific notation. default is 0 for keeping scientific notation
options(scipen=999) 

seed = 80
setwd('C:/Users/asuryav1/OneDrive - T-Mobile USA/NLP behavior analysis/model template/unsupervised clustering/tfb')
localpath = 'C:/Users/ASuryav1/temp/unsupervised clustering/tfb/'


library(mlapi)
library(R6)
library(ggplot2)  ### plotting
library(ggrepel)
library(hues)
library(data.table) ### new addition
library(reshape2) ### data manipulation
library(magrittr) ### pipes
library(plyr) ### data manipulation, load before dplyr
library(tokenizers)
library(text2vec)
library(future)
library(future.apply)
library(keras)
use_condaenv("r-tensorflow")
library(spacyr)
library(Matrix)
library(superheat)
library(cld2)
library(caret)
library(MASS)

library(lexicon)
library(tidylo)
library(stopwords)
library(umap)
library(tictoc)
library(Rtsne)
library(sentimentr)
library(tidyverse) ### piping and chaining operations. Load this package last as it is widely used and has some conflicts with other packages, especially plyr
 
source("utils_v13.R" )

write.excel <- function(x,row.names=FALSE,col.names=TRUE,...) {
  write.table(x,"clipboard",sep="\t",row.names=row.names,col.names=col.names,...)
}

# write.excel(my.df)

## models ###########

temppath = getwd()
setwd('C:/Users/ASuryav1/OneDrive - T-Mobile USA/Abhi R packages/RtextClust')
w2v.wv.filepath = paste0('models/w2v_', ## 'models/w2v_
                  ifelse(max.ngrams==2,'phrase_',''),
                  ifelse(remove_stopwords==T,'stop_','nostop_'),
                  ifelse(use_POS==T,'pos_','nopos_'),
                  wv.dim,'dim.RDS')

wv = readRDS(w2v.wv.filepath)
class(wv) <- "numeric"
setwd(temppath)
############ read data ###############
#df = read_delim('data/transcript_2020030601.txt', delim = '|')  
# df = read_delim('data/transcripts_POS_2021-06-24.txt', delim = '|') ## network outage L3 pos neg turns
df = read_delim(paste0('','case_data_cleaned_pos_v1.txt'), delim = '|') ## network outage L2 sampled for may 2021

colnames(df) = tolower(colnames(df))

# df = df[1:1000,]

######## process data ############
#### skipping pos and aggregation below as it was already done in python when data was pulled
# ## note: the function below needs spaCy python library within an anaconda environment 'r-tensorflow'. I have attached a yaml file separately to the email that can be imported into anaconda 
# process_upto_POS = function(dat){
# 
#   dat = dat%>%
#     mutate(orig.phrase = phrase)%>%
#     mutate(phrase = tolower(iconv(phrase, to = "UTF-8")) )%>% ## cleans up unwanted characters
#     mutate(phrase = str_trim(str_replace_all(phrase, '\\s+', ' ')))%>%
#     group_by(sourcemediaid)%>%
#     arrange(endoffset, .by_group = TRUE)%>%
#     # summarize(phrase = str_c( phrase, collapse = ' ')  )%>%
#     ungroup()%>%
#     mutate(phrase = preprocess.text.fn(phrase))%>%
#     mutate(cleaned.phrase = phrase)%>%
#     group_by(sourcemediaid)%>%
#     mutate(language = cld2::detect_language(str_c( phrase, collapse = ' '), lang_code = FALSE))%>%
#     ungroup() %>%
#     filter(str_count(phrase, '\\w+') > 1 & language == 'ENGLISH')
#  
#   if(use_POS == T) {
#    ### watch for RAM usage for the function below. If RAM is maxing out and disk usage goes up, then the function will slow down drastically
#   dat$phrase = suppressMessages(suppressWarnings(
#     attach_POS2(dat$phrase, python.env = 'r-tensorflow', chunksize = 50000 )  ))
#   }
#   
#   print(paste('chunk ', ctr, " has been processed"))
#   ctr <<- ctr+1
#   
#   return(dat)
# }
# 
# tic()
# chunk_size = 1e6 ## this number along with the chunk size of the attach_POS2 function in 'process_upto_POS' controls the RAM usage. Adjust the chunk size in the attach_POS2 function first before this one
# ctr = 1
# chunk_arr = (1:nrow(df))%/%( nrow(df) / (ceiling(nrow(df)/chunk_size)) + 0.01)
# df = lapply(split(df, chunk_arr ) , process_upto_POS)
# df = data.table::rbindlist(df, use.names = T)
# toc()
# 
# df$text_length = sapply(str_split(df$phrase, '\\s+'), length)
# # saveRDS(df, 'data/df.rds')
# # df = readRDS( 'data/df.rds')
# 




# traindf = df%>%
#     mutate(startoffset = as.numeric(startoffset),
#            endoffset = as.numeric(endoffset))%>%
#   group_by(sourcemediaid)%>%
#  # arrange(endoffset, .by_group = TRUE)%>%
#   summarize(txt = paste0(phrase , collapse = ' '),
#             cleaned_phrase = paste0(cleaned_phrase , collapse = ' '),
#             pred_csrsatisfaction = first(pred_csrsatisfaction),
#             pred_uncarrierloyalty = first(pred_uncarrierloyalty))%>%
#   ungroup()%>%
#   distinct(txt, .keep_all = TRUE)

# rm(df)

## use this if you want to keep it at a turn level instead of call level
traindf = df%>%
rename(txt = phrase)%>%
  distinct(txt, .keep_all = TRUE)#%>%
 # filter(row_length >=5)
  #sample_frac(0.3) ## the dataset will be large at turn-level. so sample some turns
 

########################################################################################
### If you want to change the word embeddings, either use the 'generate word vectors' file to generate a new set of word embeddings or use your own (correctly formatted) 
train.embeddings = list()
train.embeddings$word_vectors = wv
colnames(train.embeddings$word_vectors) = str_replace_all(colnames(train.embeddings$word_vectors), '^(V)([0-9]+)$', 'X\\2') ## some customization of column names to match the rest of the code and functions below
train.embeddings$tokens = tokenize_regex(traindf$txt) 

temp = get_embeddings(train.embeddings$tokens, ngram.min = 1,
                      ngram.max = max.ngrams, ngram.sep = '~', apply.tfidf = T, tfidf.norm = 'l1') 
train.embeddings$dtm = temp$word_dtm
train.embeddings$vocab = temp$vocab
train.embeddings$model.tfidf = temp$model_tfidf

train.embeddings$doc_vectors = get_doc_dtm(train.embeddings$dtm, train.embeddings$word_vectors, svd=F, min=F, max=F)
colnames(train.embeddings$doc_vectors) =  str_c('X', 1:ncol(train.embeddings$doc_vectors)) ## some customization of column names to match the rest of the code and functions below

traindf$txt_length = unlist(lapply(train.embeddings$tokens,function(x) length(x) ) )

########################################################################################
set.seed(seed)
## get a tsne matrix for plotting clustering results. This will be used later in the code. may take a while  #######
train.rtsne = create.rtsne.matrix(train.embeddings$doc_vectors, number.of.samples = 10000, rtsne.perplexity=50, seed = 80)

########## get seed words using hclust #############################

## you can try different combinations of nclust and keywords.per.clust. the code below splits the data into nclust clusters and picks top keywords.per.clust keywords in each cluster as seed words 
nclust = 100
keywords.per.clust=3


set.seed(seed)
random.indices = sample(seq(1,nrow(train.embeddings$doc_vectors),1),
                        min(nrow(train.embeddings$doc_vectors),10000), replace = F )
doc.vectors.subset = as.matrix(train.embeddings$doc_vectors[random.indices,])
distMatrix <- dist(doc.vectors.subset, method="euclidean")

clust <- hclust(distMatrix,method="ward.D")
plot(clust, cex=0.9, hang=-1)
rect.hclust(clust, k=nclust)
groups<-cutree(clust, k=nclust)


train.wordlist = get_important_words(train.embeddings$tokens[random.indices],
                                groups,
                                ngram.min = 1, ngram.max = max.ngrams, ngram.sep = '~')

temp = train.wordlist%>%
  filter(log_odds >= 1.5 )  

if (use_POS == T) { ### keep only nouns
  temp = temp%>%
    filter(str_count(term, '_NOUN')/str_count(term, '_[A-Z]+') == 1) 
}


seed.words.orig = temp %>% 
  group_by(term)%>%
  filter(log_odds == max(log_odds)) %>%
  ungroup()%>% 
  group_by(y)%>%
  top_n(keywords.per.clust,log_odds)%>% 
  slice(seq_len(keywords.per.clust))%>%
  ungroup()%>%
  group_by(y)%>%
  mutate(reasons = str_c('cr_',first(term, order_by = -log_odds)))%>%
  ungroup()%>%
  mutate(type = 'unsupervised')%>%
  rename(words = term)%>%
 select(words, reasons, type )

## exclude any unwanted categories. this can be done at the beginning, or after checking the results of the first run, eliminating some unwanted labels and rerunning
seed.words.orig = seed.words.orig%>%
  filter(!reasons %in% c(
    '	cr_recipient_NOUN'
  ))

seed.words.orig = seed.words.orig%>%
  filter(!words %in% c(''
  ))



########### clustering algorithm ################ 

control.param = list(
  'num.iter' = 30,
  'max.keywords.per.topic' = 10,
  'alpha' = 0.9, ### 0.9 alpha (between 0 to 1) controls how fast cluster centers move as new keywords are introduced. alpha 0 = fast movement, higher chance of losing the original meaning, alpha = 1 slower drift, original meaning is better retained. imagine mixing black and white colors. black = old keywords, white = new keywords. the mixture of the two colors is the new cluster center. alpha = 1 gives black, alpha = zero gives white, and alpha between 0 and 1 gives shades of grey
  'beta' = 0.1, ### beta (between 0 to 1) controls how much the cluster centers drift away from the starting point. imagine an elastic band tied between the starting cluster center and the new cluster center. beta close to 1 means the elastic band is very strong and the new center wont drift too much from the starting point. beta close to 0 means the elastic band is very weak and the new center can drift freely away from the starting center
  'cluster.merge.threshold' = 0.9, ### 0.95, ranges from 0 to 1. closer to 1 means two clusters have to be very similar to be merged into one cluster. closer to 0 means clusters that are farther away from each other can also merge rapidly 
  
  'quality.threshold.to.add.keywords' = 0.95, ### if current cluster quality is > 0.95*max( cluster quality over all iterations) then add keyword one by one
  'quality.threshold.to.remove.keywords' = 0.5, ### 0.5 , if current cluster quality is < 0.5*max( cluster quality over all iterations) then remove keyword one by one (tries to reduce the influence of the cluster as it is poor quality). between add and remove thresholds, the current number of keywords is kept as-is
  
  'silhouette.threshold' = 0, ### clusters below this silhouette threshold are considered for elimination every  'remove.silh.every.niter' iterations
  'remove.silh.every.niter' = 5, ### 100, more agressive removal will give better clusters, but may eliminate subtopics that have higher overlap with other clusters
  'silh.clusters.to.remove' = 1, ### how many clusters to remove every n iterations
  'fast.silhouette.calc' = T, ### fast.silhouette.calc = T uses an approximate silhouette calculation (dissimilarity between docs and cluster centers). fast.silhouette.calc = F uses sampled inter-document distances, which can be slower for larger datasets 
  
  'cluster.vol.threshold.pc' = 0.01 ## if cluster volume is < 1% of total, then eliminate it
)


clusters = cluster.reason.vectors(traindf, 
                                  train.embeddings$dtm, 
                                  train.embeddings$word_vectors, 
                                  train.embeddings$doc_vectors, 
                                  rw_stopw,
                                  seed.words.orig,
                                  control.param
)

reason.vectors = clusters$fnrv
orig.reason.vectors = clusters$fnorigrv
cluster.center.progress = clusters$cluster.center.progress
seed.words.iter = clusters$word.list
seed.words = clusters$seed.words


#### quality plots #####################################################
temp = seed.words.iter%>%
  group_by(reasons2,iteration)%>%
  summarize(quality = mean(quality, na.rm = T), silh = mean(silh,na.rm=T))%>%
  ungroup()%>%
  group_by(iteration)%>%
  summarize(quality = mean(quality,na.rm=T), silh = mean(silh, na.rm=T))%>%
  ungroup()

ggplot(temp, aes(iteration,quality)) + geom_line() + geom_point() 



############## cluster center progress plots ################################

temp = cluster.center.progress%>%
  mutate(reasons = str_replace_all(reasons2,'[0-9]+$','') )%>%
  rename(iteration = iter)

seed.words.iter1 = seed.words.iter%>%
  left_join(temp, by = c('iteration','reasons','reasons2'))


cluster.center.progress$reasons = str_replace_all(cluster.center.progress$reasons2,'[0-9]+$','')
temp1 = seed.words.iter1%>%
  group_by(reasons,reasons2)%>%
  filter(iteration==max(iteration))%>%
  arrange(metric, .by_group = TRUE)%>%
  summarize(newwords = str_c(unique(words), collapse = ','), iteration = max(iteration))%>%
  ungroup()

temp2 = seed.words%>%
  group_by(reasons,reasons2)%>%
  summarize(orig.words = str_c(unique(words), collapse = ','))%>%
  ungroup() 

temp3 = temp1%>%left_join(temp2, by = 'reasons2')

cluster.center.progress1 = cluster.center.progress%>%
  left_join(temp3, by = 'reasons2')%>%
  filter(iter<=iteration)

ggplot(cluster.center.progress1) + geom_line(aes(iter, improvement, col = as.factor(reasons2))) +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

###########   keywords summary  ##############################

temp = cluster.center.progress%>%
  mutate(reasons = str_replace_all(reasons2,'[0-9]+$','') )%>%
  rename(iteration = iter)

seed.words.iter1 = seed.words.iter%>%
  left_join(temp, by = c('iteration','reasons','reasons2'))

#1. Get the seed words of the last iteration
#2. Group by words

keyword.summary = seed.words.iter1%>%
  group_by(reasons,reasons2)%>%
  filter(iteration== max(seed.words.iter1$iteration) & iteration != 1)%>%
  ungroup()%>%
  group_by(words)%>%
  mutate(repeatword.rn = row_number() , repeatword.counts = n() )%>%
  ungroup()%>% 
  group_by(reasons,reasons2)%>%
  arrange(desc(metric), .by_group = TRUE)%>%
  summarize(newwords = str_c(words, collapse = ','),
            wordcount = length(words),
            reasons3 = first(str_c(words, ifelse(repeatword.counts> 1,repeatword.rn, '') ) ),
            weights = str_c(round(metric,3) , collapse = ','),
            iteration = str_c(unique(iteration), collapse = ','))%>%
  ungroup() %>%
  left_join(seed.words%>%group_by(reasons2)%>%summarize(words = paste0(words,collapse=','))%>%ungroup(), by = 'reasons2')

#Sys.time() - tim


# ################# plot silhouette ###################################
# final.silh = calculate.silhouette(reason.vectors, train.embeddings$doc_vectors, keyword.summary,
#                                   plot.silhouette = T,  fast.silhouette.calc = F  )

##################### predictions ###########################
predict.train = predict.clusters(keyword.summary, 
                                 traindf, 
                                 train.embeddings$word_vectors, 
                                 train.embeddings$doc_vectors, 
                                 train.embeddings$tokens
)

traindf1 = predict.train$predictions


# saveRDS(traindf, 'traindf.rds')
# saveRDS(traindf1, 'traindf1.rds')
# saveRDS(keyword.summary, 'keyword_summary.rds')
# saveRDS(train.embeddings, 'train_embeddings.rds')
######## plots ###########################

# prop.table(table(traindf1$top1class))

### plot predicted classes

temp_ks = keyword.summary%>%
  select(reasons3, newwords, weights)%>%
  mutate(reasons3 = str_remove_all(reasons3,'[0-9]+$|[_A-Z]+')) %>%
  separate_rows(newwords,weights,sep = ',')%>%
  mutate(weights = as.numeric(weights))%>%
  mutate(reasons3 = str_replace_all(reasons3,'~','.') )%>%
mutate(newwords = str_remove_all(newwords,'[0-9]+$|[_A-Z]+')) %>%
  mutate(newwords = str_replace_all(newwords,'~','.') )%>%
  separate_rows(newwords,sep = '\\.')


temp_ks1 = temp_ks%>%
  group_by(reasons3,newwords)%>%
  summarize(weights = max(weights))%>%
  ungroup()%>%
  arrange(desc(weights))%>%
  group_by(reasons3)%>%
  top_n(3,weights)%>%
  ungroup()%>%
  group_by(reasons3)%>%
  summarize(newwords = paste0(newwords,collapse = ', '))%>%
  ungroup()%>%
  rename(top1class = reasons3)
  
traindf2 = traindf1 %>%
  mutate(top1class = str_remove_all(top1class,'[0-9]+$|[_A-Z]+'))%>%
  left_join(temp_ks1, by = 'top1class')%>%
  group_by(newwords)%>%
  mutate(
    # category_csat_pc = round(sum(pred_csrsatisfaction <=7, na.rm = T)/n()*100,2),
    # category_csat = round(mean(pred_csrsatisfaction, na.rm = T),2),
         vol_counts = n(), vol_pc = n()/nrow(traindf1) )%>%
  ungroup()#%>%
 # mutate(newwords = str_c(newwords,'(',csat_pc,'%)'))%>%
 # mutate(top1class = str_c(top1class,'(',csat_pc,'%)'))

traindf2$newwords[traindf2$vol_pc <= 0.01] = NA
traindf2$top1class[traindf2$vol_pc <= 0.01] = NA

temp = traindf2%>%
  filter(!is.na(top1class ))%>%
  group_by(top1class,newwords)%>%
  summarize(vol_pc = first(vol_pc))%>%
  ungroup()
ggplot(temp) + geom_bar(aes(reorder(newwords,vol_pc), vol_pc),stat = 'identity') + 
   coord_flip()
  
###### data for slides #####
temp = traindf2%>%select(newwords,vol_pc)%>%distinct()%>%na.omit()
write.excel(temp)

#########################

# temp =   traindf2$top1class
temp =   traindf2%>%
  mutate(newwords = stringr::str_wrap(newwords, width = 25))%>%
  pull(newwords)


# temp =   traindf1%>%
#   mutate(reasons3 = str_remove_all(top1class,'^sim'))%>%
#   left_join(keyword.summary%>%select(reasons3, newwords)%>%
#               mutate(newwords = str_replace_all(newwords,'[_A-Z]+|,',' ')),
#             by = 'reasons3')%>%
#   mutate(newwords = stringr::str_wrap(newwords, width = 25))%>%
#   mutate(newwords = str_replace_all(newwords,' ',','))%>%
#   pull(newwords)

# temp1 =   traindf1%>%
# #  mutate(top1class = str_remove_all(top1class,'[0-9]+$|[_A-Z]+'))%>%
#  # mutate(top1class = str_replace_all(top1class,'~','.'))%>%
#   group_by(top1class)%>%
#   mutate(csat_pc = round(sum(pred_csrsatisfaction == 0)/n()*100,2) )%>%
#   ungroup()%>%
#   rename(reasons3 = top1class )
# 
# temp2 = keyword.summary%>%select(reasons3, newwords)#%>%
# #  mutate(reasons3 = str_remove_all(reasons3,'[0-9]+$|[_A-Z]+'))%>%
#  # mutate(reasons3 = str_replace_all(reasons3,'~','.'))%>%
#   # mutate(newwords = str_remove_all(newwords,'[0-9]+$|[_A-Z]+'))%>%
#   # mutate(newwords = str_replace_all(newwords,'~','.')) 
#   
#   
#  
# temp3 = temp1%>%
#   left_join(temp2,  by = 'reasons3')%>%
#   mutate(newwords = stringr::str_wrap(newwords, width = 25))%>%
#   mutate(newwords = str_replace_all(newwords,' ',','))%>%
#   pull(newwords)


# # plot.cluster.results(test.rtsne$rtsne_out,  test.rtsne$random.indices, temp)
# plot.cluster.results(train.rtsne$rtsne_out,  train.rtsne$random.indices, temp)
# 
# # plot.umap.cluster.results(test.umap,  umap.test.random.indices, temp)
# plot.umap.cluster.results(train.umap,  umap.train.random.indices, temp)
# 


library(ggridges)
theme_set(theme_minimal())

fndf = traindf2[train.rtsne$random.indices,]


tsne_plot <- data.frame(x = train.rtsne$rtsne_out$Y[,1],
                        desc = fndf$newwords,
                        vol_pc = fndf$vol_pc#,
                        # csat_pc = fndf$category_csat_pc,
                        # csat = fndf$category_csat
                        
)%>%na.omit()%>%
  group_by(desc)%>%
  mutate(d = median(x) )%>%
  ungroup() 

# plotdf = tsne_plot%>%select(desc,vol_pc,csat_pc, csat)%>%distinct()
plotdf = tsne_plot%>%select(desc,vol_pc)%>%distinct()


# write.excel(plotdf)

p1 = ggplot(plotdf,aes(reorder(desc,vol_pc), vol_pc)) + 
  geom_bar(stat = 'identity') + 
  geom_text(label = plotdf$vol_pc, vjust= 0, hjust = 0) + 
  coord_flip() + 
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank(),
        legend.position = "none")

# p2 = ggplot(plotdf) + 
#   geom_bar(aes(reorder(desc,vol_pc), csat_pc),stat = 'identity') + 
#   coord_flip() +
#   theme(axis.title.y=element_blank(),
#         axis.text.y=element_blank(),
#         axis.ticks.y=element_blank())

p3 = ggplot(tsne_plot, aes(x = x, y = reorder(desc,vol_pc), fill = ..x.. )) +
  # geom_density_ridges(aes(fill = desc), scale = 3, alpha = 0.7) + ## , rel_min_height = 0.01
  geom_density_ridges_gradient(scale = 2,  alpha = 0.7) +
  viridis::scale_fill_viridis(name = "Temp. [F]", option = "C") +
  theme(axis.title.y=element_blank(),
        text = element_text(size=20),
        #axis.text.y=element_blank(),
        #axis.ticks.y=element_blank(),
        legend.position = "none")

ggsave(filename = 'img2.png', width = 10, height = 10, device='png', dpi=700)



########## inspect calls
temp = traindf2%>%
  select(-cleaned_phrase,-txt,-txt_clean)
write_delim(temp,'C:/Users/ASuryav1/temp/L3 satisfaction 20210624/cluster_results_OutageL3_transcripts_POS_2021-06-24.txt', delim = '|')


  temp = df%>%
  select(-phrase)%>%
  inner_join(traindf2%>%filter(str_detect(newwords,'browser'))%>%
               select(sourcemediaid, newwords, category_csat, vol_pc), by = 'sourcemediaid')%>%
  mutate(rn = row_number())%>%
  select(rn,sourcemediaid, party, cleaned_phrase, pred_csrsatisfaction, 
         category_csat, vol_pc,everything())
