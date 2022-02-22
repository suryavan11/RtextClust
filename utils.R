preprocess.text.fn <- function(x) {
  x = str_replace_all( str_to_lower(x),'[^a-z ]','' )
  x = str_replace_all( str_trim(x),'\\s+',' ' )
  return(x)
}

tokenize.fn <- function(x, stopwords.list) {
  tokens = x%>%tokenize_word_stems(stopwords =  stopwords.list ) ## stopwords::stopwords('en')) 
  tokens = lapply(tokens,function(x) x[nchar(x)>2])
  tokens = lapply(tokens,function(x) x[nchar(x)<= 20])
  tokens = lapply(tokens,function(x) x[!x %in% stopwords.list])
  return(tokens)
}

get.metric = function(fndtm,fnrv,fnwv, fndv, fnrw_stopw, cosine.threshold = 0.8, use.dist = F) {
#    browser()
  
  rn = rownames(fnrv)
  fnrv = fnrv%>%
    dplyr::select(starts_with('X'))%>%
    as.matrix()
  rownames(fnrv) = rn
  
   rw = sim2(fnrv, fnwv, method = 'cosine', norm = 'l2')  ### cosine-reason & word
  # rw = sim2( sweep(fnrv, MARGIN = 2, STATS =  as.vector(fnrw_stopw), FUN = "-"),
  #           sweep(fnwv, MARGIN = 2, STATS = as.vector(fnrw_stopw), FUN = "-"),
  #           method = 'cosine', norm = 'l2')  ### cosine-reason & word

     
  common_terms = intersect(colnames(fndtm), colnames(rw) )
  dw = fndtm[,common_terms, drop = F] 

   dr =  sim2(fndv, fnrv, method = 'cosine', norm = 'l2')
  # dr = sim2( sweep(fndv, MARGIN = 2, STATS =  as.vector(fnrw_stopw), FUN = "-"),
  #            sweep(fnrv, MARGIN = 2, STATS =  as.vector(fnrw_stopw), FUN = "-"),
  #            method = 'cosine', norm = 'l2')
  
  rw = rw[colnames(dr),colnames(dw), drop = F]   ### ensure that all matrices are lined up correctly before any matrix operation
  
  
  if (use.dist == T) {
    ### dist matrix
    
    indices =   apply(dr, 2, function(x) sample(which(x>0), min(length(which(x>0)),5000, na.rm = T)  ) )  
    dr.mod = dr
    dr.mod[,] = NA
    
    for (ind in 1:dim(dr)[2]) { 
      # print(ind)
      dd =   sim2(fndv[indices[[ind]],] ,  method = 'cosine', norm = 'l2') 
      dd.soft = 1/(2-dd) 
      dr.mod[indices[[ind]],ind] = (dd.soft %*% dr[indices[[ind]],1,drop = F] )/rowSums(dd.soft)
    }
    
  }
  # metric1 = as.matrix( (t(dw) %*% dr)/(t(dw) %*% (dr*0 + 1) )  )
  
  ### wd x dr / (denom) is the weighted average of dr similarities by the tfidf counts (the denom is essentially just a sum of the tfidf weights per word. the *0+1 part creates a matrix of ones)
  ### the above weighted average says whether a word is tied to a reason, but does not account for its ties to other reasons. multiplying by t(rw) brings that in

  if (use.dist == T) {
    metric = as.matrix( (t(dw) %*% dr.mod)/(t(dw) %*% (dr*0 + 1) ) * t(rw) )
  } else {
    metric = as.matrix( (t(dw) %*% dr)/(t(dw) %*% (dr*0 + 1) ) * t(rw) )
  }
  
 
  ##################################################################

  # word.list = as.data.frame(metric)%>%
  #   mutate(words = rownames(metric))%>%
  #   melt(id.vars = 'words', variable.name = 'reasons2', value.name = 'metric') %>%
  #   mutate(reasons = str_replace_all(reasons2,'[0-9]+$',''))%>%
  #   filter(metric>0) # %>%
  # # group_by(reasons2)%>%
  # # mutate(metric = exp(-1*row_number(desc(metric)))*metric )%>%
  # # ungroup()

  word.list = as.data.frame(metric)%>%
    mutate(words = rownames(metric))%>%
    melt(id.vars = 'words', variable.name = 'reasons2', value.name = 'metric') %>%
    mutate(reasons = str_replace_all(reasons2,'[0-9]+$',''))%>%
    filter(metric>0)
  
  if (use_POS == T) {
    word.list = word.list%>%
    filter(str_count(words, '_NOUN')/str_count(words, '_[A-Z]+') == 1) 
  }
  
  return(word.list)
  
  
}

#measure similarity between reasons
calculate.silhouette <- function(fnrv, fndv, fnkey.sum = NULL, plot.silhouette = T, fast.silhouette.calc = F) {
 # browser()
  if (!is.null(fnkey.sum)) {
    rownames(fnrv) = as.character ( data.frame('reasons2' = rownames(fnrv))%>%
                                      left_join(fnkey.sum, by = 'reasons2')%>%
                                      pull(reasons3) )
    
  }
  
  rn= rownames(fnrv)
  fnrv = fnrv%>%
    dplyr::select(starts_with('X'))%>%
    as.matrix()
  rownames(fnrv) = rn
 
  
  #browser()
  dr =  sim2(fndv, fnrv, method = 'cosine', norm = 'l2') 
  
  #find similarity between document and reason
  membership =   t(apply(dr, 1, function(x) as.numeric(x != max(x))))
  membership[membership == 0] = NA
  membership1 = colnames(dr)[apply(dr,1, which.max)]
  ## dr.dissim = acos(round(dr, 10)) / pi
  dr.dissim = round(1-dr,3)
  
  
  if (fast.silhouette.calc == T) {
    ### https://rlbarter.github.io/superheat-examples/Word2Vec/
    ### this is not the true silhouette calculation. the true silhouette calc would need a distance matrix between docs
    silh  = apply(dr.dissim * membership,1,min) - apply(dr.dissim,1,min)
    
  } else {
    
    #### alternative silh calculation
    indices =   apply(membership, 2, function(x) sample(which(is.na(x)), min(length(which(is.na(x))),5000, na.rm = T)  ) )  
    ai = as.numeric(rep(NA,nrow(dr)) )
    
    
    for (ind in 1:dim(dr)[2]) { 
      # print(ind)
      if (length(indices[[ind]]) != 0  ) {
        dd.dissim =  1- sim2(fndv[indices[[ind]],,drop=F] ,  method = 'cosine', norm = 'l2') 
        ai[  indices[[ind]]  ] = rowMeans(dd.dissim)
      }
    }
    
    bi = apply(dr.dissim * membership,1,function(x) min(x, na.rm = T)  )
    bi[bi==Inf] = NA   ### Inf are coming from short texts that are not assigned to any cluster
    
    silh =  bi - ai
    
  }
  
  #### quality/coherence calculation
  temp =  t(apply(dr, 1, function(x) as.numeric(x == max(x))))
  temp[temp==0] = NA
  qual = colSums(dr*temp, na.rm = T)  # dr*temp: max
  
  
  temp = as.matrix(dr )
  temp.silh = data.frame(membership1, silh )%>%group_by(membership1)%>%summarize(silh = mean(silh))%>%ungroup()
  temp.silh = rbind(temp.silh,
                    data.frame('membership1' = colnames(temp)[!colnames(temp) %in% unique(temp.silh$membership1)],
                               'silh' = rep(0,length(colnames(temp)[!colnames(temp) %in% unique(temp.silh$membership1)] )) 
                    ) )
  
  temp.silh = temp.silh%>%left_join(data.frame('quality' = qual, 'membership1' = names(qual)), by = 'membership1' )
  
  if (plot.silhouette == T) {
    g1 = superheat(temp, 
                   
                   # row and column clustering
                   membership.rows = membership1 ,
                   membership.cols = colnames(temp),
                   
                   # top plot: silhouette
                   yt = temp.silh$silh[match(colnames(temp) , temp.silh$membership1 )],  ### this is a little tricky. the order of yt should match the original colnames(temp) order, not the order(colnames(temp)) order
                   yt.axis.name = "Cosine\nsilhouette\nwidth",
                   yt.plot.type = "bar",
                   yt.bar.col = "grey35",
                   
                   # order of rows and columns within clusters
                   order.rows = order(membership1 ),
                   order.cols = order(colnames(temp)),
                   
                   # bottom labels
                   bottom.label.col = c("grey95", "grey80"),
                   bottom.label.text.angle = 90,
                   bottom.label.text.alignment = "right",
                   bottom.label.size = 0.28,
                   
                   # left labels
                   left.label.col = c("grey95", "grey80"),
                   left.label.text.alignment = "right",
                   #left.label.size = 0.26,
                   
                   # smooth heatmap within clusters
                   smooth.heat = T,
                   
                   # title
                   title = "(b)")
    
    g1
    
  }
  
  return(temp.silh)
  
}

get.reason.vectors <- function(fnoldrv,fnorigrv, fnwv,word.list, weigh.by.similarity = T, alpha = 0, beta = 0.8 ) {
  
 # browser()
  
  fnrv =data.frame(fnwv[rownames(fnwv) %in% word.list$words,,drop = FALSE])%>%
    mutate(words = row.names(.))%>%
    left_join(word.list, by = 'words')
  
  fnrv[,str_detect(colnames(fnrv),'^X')] = fnrv[,str_detect(colnames(fnrv),'^X')] * fnrv$sign
  
  fnrv = fnrv%>%
    dplyr::select(reasons,reasons2,metric,words,everything())%>% dplyr::select(-words, -sign)
  
  ### weighting by simlarity. w1*x1/sum(w) is calculated for each line and then sum  at the end (this rowwise calculation is weighted average and replaces mean)
  if (weigh.by.similarity == T) {
    fnrv[,str_detect(colnames(fnrv),'^X')] = fnrv[,str_detect(colnames(fnrv),'^X')]*fnrv$metric
    fnrv = fnrv%>%
      group_by(reasons,reasons2)%>%
      mutate(metric = 1/sum(metric))%>%
      ungroup()
    fnrv[,str_detect(colnames(fnrv),'^X')] = fnrv[,str_detect(colnames(fnrv),'^X')]*fnrv$metric
    fnrv = fnrv%>%dplyr::select(-metric)%>%group_by(reasons,reasons2)%>%summarize_all(funs(sum))%>%ungroup()
    rownames(fnrv) = fnrv$reasons2
    
  } else {
    
    fnrv = fnrv%>%dplyr::select(-metric)%>%group_by(reasons,reasons2)%>%summarize_all(funs(mean))%>%ungroup()
    rownames(fnrv) = fnrv$reasons2
    
  }
  
  if(is.null(fnoldrv)) {
    return(fnrv)
  } else { 
    
    rn = rownames(fnrv)
    fnrv = fnrv%>%
      dplyr::select(starts_with('X'))%>%
      as.matrix()
    rownames(fnrv) = rn
    
    rn = rownames(fnoldrv)
    fnoldrv = fnoldrv%>%
      dplyr::select(starts_with('X'))%>%
      as.matrix()
    rownames(fnoldrv) = rn
    fnoldrv = fnoldrv[rownames(fnrv),]
    
    rn = rownames(fnorigrv)
    fnorigrv = fnorigrv%>%
      dplyr::select(starts_with('X'))%>%
      as.matrix()
    rownames(fnorigrv) = rn
    fnorigrv = fnorigrv[rownames(fnrv),]
    
    
    temp = psim2(fnrv, fnorigrv,method = 'cosine', norm = 'l2') - beta
    temp[temp<0] = 0
    temp = 1-temp
    temp[temp<alpha] = alpha
    
    fnrv = fnrv * (1-temp) + fnoldrv * temp
    fnrv = data.frame(fnrv)%>%
      mutate(reasons2 = rownames(.), reasons = str_remove_all(reasons2,'[0-9]+$') )
    rownames(fnrv) = fnrv$reasons2
    
    return(fnrv)
  }
  
}

unsup.choose.keywords <- function(fndf, fndv, fnvocab, nclust = 50,keywords.per.clust=3, plot = F) {
  ################### choose seed keywords by hclust ########################
  
  fndf = traindf
  fndv = train.embeddings$doc_vectors
  fnvocab = train.embeddings$vocab
  nclust = 50
  keywords.per.clust=3
  
   set.seed(80)
  random.indices = sample(seq(1,nrow(fndv),1), min(nrow(fndv),10000), replace = F )
  doc.vectors.subset = as.matrix(fndv[random.indices,])
  distMatrix <- dist(doc.vectors.subset, method="euclidean")
  
  clust <- hclust(distMatrix,method="ward.D")
  plot(clust, cex=0.9, hang=-1)
  rect.hclust(clust, k=nclust)
  groups<-cutree(clust, k=nclust)
  
  
  train.wordlist = get_important_words(train.embeddings$tokens[random.indices],
                                       groups,
                                       ngram.min = 1, ngram.max = 2, ngram.sep = '~')
  
  
  
  temp = train.wordlist%>%
    filter(log_odds >= 2 )%>%
    filter(str_count(term, '_NOUN')/str_count(term, '_[A-Z]+') == 1) 
  
  common_terms = intersect(temp$term, rownames(train.embeddings$word_vectors) )
  temp1 = sim2(train.embeddings$word_vectors[common_terms,],rw_stopw, method = 'cosine', norm = 'l2')
  temp2 = temp%>%
    left_join(data.frame('stopw_cossim' = temp1,'term' =  common_terms), by = 'term' )
  
  seed.words.orig = temp2%>%
    filter(stopw_cossim <= 0.2)%>%
    group_by(term)%>%
    # arrange(desc(log_odds), .by_group = TRUE)%>%
    filter(log_odds == max(log_odds)) %>%
    ungroup()%>% 
    group_by(y)%>%
    top_n(3,log_odds)%>% 
    slice(seq_len(3))%>%
    ungroup()%>%
    group_by(y)%>%
    mutate(reasons = str_c('cr_',first(term, order_by = -log_odds)))%>%
    ungroup()%>%
    mutate(type = 'unsupervised')%>%
    rename(words = term)%>%
    select(words, reasons, type )
  
  
  
  
  
  set.seed(80)
  random.indices = sample(seq(1,nrow(fndv),1), min(nrow(fndv),10000), replace = F )
  doc.vectors.subset = as.matrix(fndv[random.indices,]) 
  distMatrix <- dist(doc.vectors.subset, method="euclidean")
  
  clust <- hclust(distMatrix,method="ward.D")
  plot(clust, cex=0.9, hang=-1)
  rect.hclust(clust, k=nclust)
  groups<-cutree(clust, k=nclust)
  
  if (plot==T) {
    rtsne_out <- Rtsne(doc.vectors.subset , perplexity = 50 ,check_duplicates = FALSE)
    
    tsne_plot <- data.frame(x = rtsne_out$Y[,1],
                            y = rtsne_out$Y[,2],
                            desc = groups)
    
    tsne_labels = tsne_plot%>%group_by(desc)%>%summarize(x = median(x), y = median(y))%>%ungroup()
    
    
    
    ggplot(tsne_plot%>%sample_n(5000), aes(x = x, y = y)) +
      stat_density2d(aes(fill = ..density.., alpha = 1), geom = 'tile', contour = F) +
      scale_fill_distiller(palette = 'Greys') + ###RdYlBu
      geom_point(aes(color=as.factor(as.integer(desc)) ),  size= 3, alpha = 1 ) +
      ggrepel::geom_label_repel(data = tsne_labels,
                                aes(x=x, y=y,label=desc, color = as.factor(as.integer(desc))),
                                fontface = 'bold' ) +
      scale_color_manual(values=rev(hues::iwanthue(200))) +
      theme(legend.position="none")
    
  }
  
  
  data.temp = fndf[random.indices,]
  
  
  for (i in seq_along(unique(groups) )) {
    
    print(i)
    df = data.temp%>%
      mutate(flag = ifelse( groups==i,2,1) )%>%
      group_by(flag) %>%
      summarize(txt = str_c(txt, collapse = ' ', sep = ' '))%>%
      ungroup()
    
    tokens.temp = tokenize.fn(df$txt,stopwords_longlist)
    it.temp = itoken(tokens.temp, progressbar = FALSE)
    vocab.temp <- create_vocabulary(it.temp, ngram = c(ngram_min = 1L, ngram_max = 2L), sep_ngram = "_" )
    vocab.temp <- prune_vocabulary(vocab.temp, term_count_min = 50L)
    vocab.temp = vocab.temp[vocab.temp$term %in% fnvocab$term,]
    vectorizer.temp <- vocab_vectorizer(vocab.temp)
    dtm.temp = create_dtm(it.temp, vectorizer.temp) 
    
    
    
    if(i==1) {
      seed.words.orig.temp = as.data.frame(t(as.matrix(dtm.temp)))%>%
        mutate(ratios = (`2`+1)/(`1`+1), words = rownames(.), 
               reasons = str_c('cr_', first(words[order(ratios,decreasing = T)]) ) )  %>%
        top_n(keywords.per.clust,ratios)%>% slice(seq_len(keywords.per.clust))  
      
    } else {
      seed.words.orig.temp = rbind(seed.words.orig.temp, as.data.frame(t(as.matrix(dtm.temp)))%>%
                                     mutate(ratios = (`2`+1)/(`1`+1), words = rownames(.),
                                            reasons =  str_c('cr_', first(words[order(ratios,decreasing = T)]) ) )  %>%
                                     top_n(keywords.per.clust,ratios)%>%slice(seq_len(keywords.per.clust))
      )
    }
    
  }
  
  
  seed.words.orig = seed.words.orig.temp %>%
    arrange(desc(ratios)) %>%
    dplyr::select(words,reasons)%>%
    mutate(type = 'unsupervised')
  
  return(seed.words.orig)
  
}

cluster.reason.vectors <- function(fndf, fndtm, fnwv,fndv, fnrw_stopw, orig.word.list, control.param = list() ) {
 # browser()
  if (length(control.param) == 0 ) {
    control.param = list(
      'num.iter' = 50,
      'max.keywords.per.topic' = 50,
      'alpha' = 0.9, ### alpha (between 0 to 1) controls how fast cluster centers move as new keywords are introduced. alpha 0 = fast movement, higher chance of losing the original meaning, alpha = 1 slower drift, original meaning is better retained. imagine mixing black and white colors. black = old keywords, white = new keywords. the mixture of the two colors is the new cluster center. alpha = 1 gives black, alpha = zero gives white, and alpha between 0 and 1 gives shades of grey
      'beta' = 0.1, ### beta (between 0 to 1) controls how much the cluster centers drift away from the starting point. imagine an elastic band tied between the starting cluster center and the new cluster center. beta close to 1 means the elastic band is very strong and the new center wont drift too much from the starting point. beta close to 0 means the elastic band is very weak and the new center can drift freely away from the starting center
      'cluster.merge.threshold' = 0.9, ### ranges from 0 to 1. closer to 1 means two clusters have to be very similar to be merged into one cluster. closer to 0 means clusters that are farther away from each other can also merge rapidly 
      
      'quality.threshold.to.add.keywords' = 0.95, ### if current cluster quality is > 0.95*max( cluster quality over all iterations) then add keyword one by one
      'quality.threshold.to.remove.keywords' = 0.5, ### if current cluster quality is < 0.55*max( cluster quality over all iterations) then remove keyword one by one (tries to reduce the influence of the cluster as it is poor quality). between add and remove thresholds, the current number of keywords is kept as-is
      
      'silhouette.threshold' = 0, ### clusters below this silhouette threshold are considered for elimination every  'remove.silh.every.niter' iterations
      'remove.silh.every.niter' = 4, ### more agressive removal will give better clusters, but may eliminate subtopics that have higher overlap with other clusters
      'silh.clusters.to.remove' = 1, ### how many clusters to remove every n iterations
      'fast.silhouette.calc' = F, ### fast.silhouette.calc = T uses an approximate silhouette calculation (dissimilarity between docs and cluster centers). fast.silhouette.calc = F uses sampled inter-document distances, which can be slower for larger datasets 
      'cluster.vol.threshold.pc' = 0.01 ## if cluster volume is < 1% of total, then eliminate it
      
    )
  }
  
  #browser()
  
  ### stem keywords
  seed.words = orig.word.list%>%
    separate_rows(words, sep = ',')%>%
    mutate(words = trimws(words))%>%
    group_by(reasons)%>%
    mutate(reasons2 = str_c(reasons,row_number(),sep='') )%>%
    ungroup()%>%
    separate_rows(words,sep='\\s+' )%>%
    mutate(sign = ifelse(str_detect(words,'^-'),-1,1) )# %>%
   # mutate(words = str_replace_all(words, c('_'=' ','-'='' ) )  )%>%as.data.frame() 
  
  # ### stem only for seeded keywords (unsupervised keywords are already stemmed)
  # if(seed.words$type == 'seeded') {
  #   seed.words$words =  unlist(lapply(seed.words$words%>%tokenize_word_stems(), function(x) str_c(x,collapse = '_') ))
  # } else {
  #   # seed.words$words =  unlist(lapply(seed.words$words%>%tokenize_words(), function(x) str_c(x,collapse = '_') ))
  # }
  
  print(paste0('words not found in the text: ',
               paste0(seed.words$words[!str_replace_all(seed.words$words,'-','') %in% colnames(fndtm)], collapse = ',') ) )
  seed.words = seed.words[str_replace_all(seed.words$words,'-','') %in% colnames(fndtm),]
  
  
  word.list = seed.words%>%
    mutate(metric = 1, iteration = 1)%>%
    dplyr::select(-type)
  
  fnrv = get.reason.vectors(fnoldrv = NULL, fnorigrv = NULL, fnwv, word.list,
                            weigh.by.similarity = T, alpha = 0, beta = 0 ) 
  
  fnorigrv = fnrv
  
  silh = calculate.silhouette(fnrv, fndv,  plot.silhouette = F , 
                              fast.silhouette.calc = control.param$fast.silhouette.calc)
  word.list = word.list%>%
    left_join(silh%>%rename(reasons2 = membership1), by = 'reasons2' )
  
  word.list = word.list%>%
    dplyr::select(words,sign, reasons, reasons2, metric, iteration, silh, quality)  
  
  
  
  
  ########### iterate
  
  for (iter in seq(2,control.param$num.iter,1)) {
    
    print(paste0('iter: ',iter))
    
    if (iter == 2) {
      print('starting point' )
      print(paste0('unique words: ',length(unique(word.list$words) ) ) )
      print(paste0('unique reasons: ',length(unique( word.list$reasons ) ) ))
      print(paste0('unique reasons2: ',length(unique(str_c(word.list$reasons,word.list$reasons2,sep='') ) ) ))
    }
    
    cosine.distances = get.metric(fndtm, fnrv, fnwv, fndv,fnrw_stopw, cosine.threshold = 0.5,use.dist = F) 
    
    temp = word.list%>%
      group_by(reasons2, iteration)%>%
      summarize(quality = unique(quality), wordcount = length(words) ) %>%
      ungroup()%>%
      group_by(reasons2)%>%
      summarize(flag = ifelse( max(quality) * control.param$quality.threshold.to.add.keywords >=  last(quality,order_by = iteration)  ,
                               ifelse(max(quality) * control.param$quality.threshold.to.remove.keywords >= last(quality,order_by = iteration), 
                               max(last(wordcount,order_by = iteration)-1,0), last(wordcount,order_by = iteration) ),
                               min(last(wordcount,order_by = iteration) + 1 ,control.param$max.keywords.per.topic)        ) )%>%
      ungroup()
    
    
    
    cosine.distances = cosine.distances%>%
      left_join(temp, by = 'reasons2')%>%
      group_by(reasons,reasons2)%>%
      filter( rank(desc(metric), ties.method="first")<= flag  )%>%
      ungroup()
    
    cosine.distances$sign = 1
    
    
    # ########## adjust keyword importance based on conflicting selections in other categories
    # cosine.distances = cosine.distances%>%group_by(words)%>%top_n(1,metric)%>%ungroup()
    
    
    # ######### restrict keywords based on deviation from average metric
    # cosine.distances = cosine.distances%>%
    #   group_by(reasons2)%>%
    #   filter(!(mean(metric)-metric) > 1*sd(metric)  )%>%
    #   ungroup()
    
    
    #############################################################################
    
    fnrv = get.reason.vectors(fnrv,fnorigrv , fnwv,cosine.distances, 
                              weigh.by.similarity = T, alpha = control.param$alpha, beta = control.param$beta  ### alpha changed to 0 from 0.9 temporarily to explore exponential weighting
                              
    )  ## alpha 0 = use new vectors, 1 = pin to old vectors
    
    
    ########## filter similar keywords
    rn = rownames(fnrv)
    fnrv.temp = as.matrix(fnrv%>%dplyr::select(starts_with('X')))
    rownames(fnrv.temp) = rn
    
    temp = t(sim2(fnrv.temp,fnrv.temp, method = 'cosine', norm = 'l2'))
    temp[lower.tri(temp, diag = FALSE)] <- NA
    temp1 = data.frame(temp)%>%
      mutate(reasonsA = rownames(.))%>%
      melt(id.vars = 'reasonsA', variable.name = 'reasonsB', value.name = 'similarity')%>%
      filter(reasonsA != reasonsB & similarity >=control.param$cluster.merge.threshold )%>%as.data.frame()
    
    for (p in seq_along(temp1$reasonsA)) {
      temp1$reasonsA[temp1$reasonsB == temp1$reasonsB[p]] = temp1$reasonsA[p]
    }
    
    
    remove.list = as.character(unique(temp1$reasonsB)[! unique(temp1$reasonsB) %in% unique(temp1$reasonsA)])
    
 
    silh = calculate.silhouette(fnrv, fndv,  plot.silhouette = F, 
                                fast.silhouette.calc = control.param$fast.silhouette.calc )
    
    cosine.distances = cosine.distances%>%
      left_join(silh%>%rename(reasons2 = membership1), by = 'reasons2' )
    
    if(iter %% control.param$remove.silh.every.niter == 0) {
      temp = as.character(silh%>%filter(silh < control.param$silhouette.threshold)%>%
                            top_n(control.param$silh.clusters.to.remove,desc( silh ))%>%
                            slice(control.param$silh.clusters.to.remove)%>%
                            pull(membership1)
      )
      print(temp)
      remove.list = unique(c(remove.list, temp ) )
    }
    
    cosine.distances = cosine.distances[!cosine.distances$reasons2 %in% remove.list,]
    fnrv = fnrv[!fnrv$reasons2 %in% remove.list,]
    
 
    
    
    #########################################################
    
    
    cosine.distances = cosine.distances%>%
      mutate(iteration = iter)%>%
      dplyr::select(words,sign,reasons, reasons2, metric, iteration, silh, quality) ## doc.similarity
    
    rn = rownames(fnrv)
    new1 = as.matrix(fnrv%>%dplyr::select(starts_with('X')))
    rownames(new1) = rn
    
    rn = rownames(fnorigrv)
    orig1 = as.matrix(fnorigrv%>%dplyr::select(starts_with('X')))
    rownames(orig1) = rn
    
    orig1 = orig1[rownames(orig1) %in% rownames(new1),,drop=F]
    
    ### print progress
    if (iter == 2) {
      
      cluster.center.progress = data.frame('improvement' = psim2(orig1, new1, method = 'cosine', norm = 'l2'), 'iter' = iter)
      cluster.center.progress$reasons2 = rownames(cluster.center.progress)
      
    } else {
      temp = data.frame('improvement' = psim2(orig1, new1, method = 'cosine', norm = 'l2'),  'iter' = iter)
      temp$reasons2 = rownames(temp) 
      cluster.center.progress = rbind(cluster.center.progress,temp )
      
    }
    
    word.list = rbind(word.list,cosine.distances)
    
    
    temp = word.list%>%filter(iteration == iter)
    print(paste0('unique words: ',length(unique(temp$words) ) )) 
    print(paste0('unique reasons: ',length(unique( temp$reasons ) ) ))
    print(paste0('unique reasons2: ',length(unique(str_c(temp$reasons,temp$reasons2,sep='') ) ) ))
    
    
  }
  
  return.list = list('fndf' = fndf,
                     'fnrv' = fnrv,
                     'fnorigrv' = fnorigrv,
                     'cluster.center.progress' = cluster.center.progress,
                     'word.list' = word.list,
                     'seed.words' = seed.words )
  
  
}

plot.cluster.results <- function(rtsne_out,  random.indices, fnlabels ) {
  
   #browser()
  
  tsne_plot <- data.frame(x = rtsne_out$Y[,1],
                          y = rtsne_out$Y[,2],
                          desc = fnlabels[random.indices]
  )
  
  
  
  #tsne_labels = tsne_plot%>%group_by(desc)%>%summarize(x = median(x), y = median(y))%>%ungroup()
  
  tsne_plot = tsne_plot%>%group_by(desc)%>%
    mutate(d = ((x-median(x))^2 + (y-median(y))^2 )^0.5 )%>%
    mutate(desc1 = as.factor(ifelse(d == min(d),as.character(desc),'') ) )%>%
    ungroup() 
  
  #getting the convex hull of each unique point set
 

  
  # g1 = ggplot(tsne_plot%>%sample_n(min(length(random.indices),nrow(tsne_plot)) ), aes(x = x, y = y)) +
  #   stat_density2d(aes(fill = ..density.., alpha = 1), geom = 'tile', contour = F) +
  #   scale_fill_distiller(palette = 'Greys') + ###RdYlBu
  #   geom_point(aes(color=as.factor(as.integer(desc)) ),  size= 3, alpha = 1 ) +
  #   ggrepel::geom_label_repel(data = tsne_labels,
  #                             aes(x=x, y=y,label=desc, color = as.factor(as.integer(desc))),
  #                             fontface = 'bold' ) +
  #   scale_color_manual(values= rev(hues::iwanthue(200)) ) +
  #   theme(legend.position="none")
  
  g1 = ggplot(tsne_plot%>%sample_n(min(length(random.indices),nrow(tsne_plot)) ), aes(x = x, y = y)) +
    stat_density2d(aes(fill = ..density.., alpha = 1), geom = 'tile', contour = F) +
   scale_fill_distiller(palette = 'Greys') + ###RdYlBu
    geom_point(aes(color=as.factor(as.integer(desc)) ),  size= 3, alpha = 1 ) +
    ggrepel::geom_label_repel( aes(x=x, y=y,label=desc1, color = as.factor(as.integer(desc))),
                              fontface = 'bold',
                              label.size = NA,  
                              size = 4,
                              label.padding=.1, 
                              na.rm=TRUE,
                              fill = alpha(c("white"),0.95),
                              force = 1,
                             # xlim = c(NA, NA), ylim = c(NA, NA)
                             # hjust = "outward",
                             # direction = "y",
                             # seed = 57,
                             # min.segment.length = 100
                             ) +
    scale_color_manual(values= rev(hues::iwanthue(200)) ) +
    theme(legend.position="none")
  
  #g1
  
  return(g1)
  
}

plot.umap.cluster.results <- function(umap_out,  random.indices, fnlabels ) {
  
  
  
  tsne_plot <- data.frame(x = umap_out$layout[,1],
                          y = umap_out$layout[,2],
                          desc = fnlabels[random.indices]
  )
  
  tsne_plot = tsne_plot%>%group_by(desc)%>%
    mutate(d = ((x-median(x))^2 + (y-median(y))^2 )^0.5 )%>%
    mutate(desc1 = as.factor(ifelse(d == min(d),as.character(desc),'') ) )%>%
    ungroup() 
  
 # tsne_labels = tsne_plot%>%group_by(desc)%>%summarize(x = median(x), y = median(y))%>%ungroup()
  
  
  g1 = ggplot(tsne_plot%>%sample_n(min(length(random.indices),nrow(tsne_plot)) ), aes(x = x, y = y)) +
    stat_density2d(aes(fill = ..density.., alpha = 1), geom = 'tile', contour = F) +
    scale_fill_distiller(palette = 'Greys') + ###RdYlBu
    geom_point(aes(color=as.factor(as.integer(desc)) ),  size= 3, alpha = 1 ) +
    ggrepel::geom_label_repel( aes(x=x, y=y,label=desc1, color = as.factor(as.integer(desc))),
                               fontface = 'bold',
                               label.size = NA,  
                               size = 4,
                               label.padding=.1, 
                               na.rm=TRUE,
                               fill = alpha(c("white"),0.95),
                               force = 1,
                               # xlim = c(NA, NA), ylim = c(NA, NA)
                               # hjust = "outward",
                               # direction = "y",
                               # seed = 57,
                               # min.segment.length = 100
    ) +
    scale_color_manual(values= rev(hues::iwanthue(200)) ) +
    theme(legend.position="none")
  
  
  return(g1)
  
}

predict.clusters <- function(fnkey.sum, fndf, fnwv, fndv, fntokens) {
  # browser()
  temp = str_split(fnkey.sum$newwords,',')
  temp =   lapply(temp,function(x) x[x %in% row.names(fnwv)])
  temp1 = str_split(fnkey.sum$weights,',')
  temp1 =   mapply(function(x,y) as.numeric(y[x %in% row.names(fnwv)]),temp,temp1, SIMPLIFY = F)
  newword.avg.vectors = t(mapply(function(x,y) Matrix::colMeans(fnwv[x,,drop=F]*y, na.rm = T) ,
                                 temp,temp1
  ))
  rownames(newword.avg.vectors) = fnkey.sum$reasons3 #paste0('sim',fnkey.sum$reasons3)
  
  
  cos.sim = data.frame( as.matrix(sim2(fndv,newword.avg.vectors , method = c("cosine"), norm = c("l2")) ) )
  
  cos.sim.class = t(apply(cos.sim, 1, function(x) names(x)[order(x, decreasing = T)]))
  cos.sim.class = cos.sim.class[,1:ifelse(ncol(cos.sim)>3,3,ncol(cos.sim)),drop=F]
  colnames(cos.sim.class) = paste0('top',1:ncol(cos.sim.class),'class',sep='')
  
  cos.sim.pred = t(apply(cos.sim, 1, function(x)  sort(x, decreasing = T) ))
  cos.sim.pred = cos.sim.pred[,1:ifelse(ncol(cos.sim)>3,3,ncol(cos.sim)),drop=F]
  colnames(cos.sim.pred) = paste0('top',1:ncol(cos.sim.pred),'pred',sep='')
  
  
  txt_clean =  lapply(fntokens,function(x) str_c(x, collapse = ' ')) 
  txt_clean = lapply(txt_clean, function(x) ifelse(identical(x,character(0)),' ',x) )
  txt_clean = unlist(txt_clean)
  # txt_length = unlist(lapply(fntokens,function(x) length(x) ) )
  
  data4 = cbind(fndf,  txt_clean, cos.sim.class, cos.sim.pred)
  
  fnkey.sum = fnkey.sum%>%
    left_join(data4%>%
               # mutate(top1class = str_remove_all(top1class,'^sim'))%>%
                group_by(top1class)%>%
                summarize(counts = n())%>%ungroup()%>%
                rename(reasons3 = top1class), by = 'reasons3' )
  
  return.list = list('predictions' = data4,
                     'keyword.summary' = fnkey.sum,
                     'reason.vectors' = newword.avg.vectors
                     )
  
  return(return.list)
  
}

get_important_words = function(tokens, y, ngram.min = 1, ngram.max = 1, ngram.sep="_") {
  ######## bind log odds ###########
  
  # y = (1:length(tokens)) %/% (round(length(tokens)/3))
  
  wordlist = lapply(levels(as.factor(y)), function(x) {
    print(x)
    it <- itoken(tokens[y==x], progressbar = FALSE)
    vocab <- create_vocabulary(it, ngram = c(ngram_min = ngram.min, ngram_max = ngram.max),
                               sep_ngram = ngram.sep)%>%mutate(y = x)
  }
  )
  
  ### this function is from tidylo version 10.1.0.900. This is an early version and the function seems to work fine. The later versions have parameters that need to be set
  abhi_bind_log_odds <-function (tbl, set, feature, n) 
  {
    set <- enquo(set)
    feature <- enquo(feature)
    n_col <- enquo(n)
    grouping <- group_vars(tbl)
    tbl <- ungroup(tbl)
    freq1_df <- count(tbl, !!feature, wt = !!n_col)
    freq1_df <- rename(freq1_df, freq1 = n)
    freq2_df <- count(tbl, !!set, wt = !!n_col)
    freq2_df <- rename(freq2_df, freq2 = n)
    df_joined <- left_join(tbl, freq1_df, by = rlang::as_name(feature))
    df_joined <- mutate(df_joined, freqnotthem = freq1 - !!n_col)
    df_joined <- mutate(df_joined, total = sum(!!n_col))
    df_joined <- left_join(df_joined, freq2_df, by = rlang::as_name(set))
    df_joined <- mutate(df_joined, freq2notthem = total - freq2, 
                        l1them = (!!n_col + freq1)/((total + freq2) - (!!n_col + 
                                                                         freq1)), l2notthem = (freqnotthem + freq1)/((total + 
                                                                                                                        freq2notthem) - (freqnotthem + freq1)), sigma2 = 1/(!!n_col + 
                                                                                                                                                                              freq1) + 1/(freqnotthem + freq1), log_odds = (log(l1them) - 
                                                                                                                                                                                                                              log(l2notthem))/sqrt(sigma2))
    tbl$log_odds <- df_joined$log_odds
    if (!is_empty(grouping)) {
      tbl <- group_by(tbl, !!sym(grouping))
    }
    tbl
  }
  
  
  
  
  wordlist = bind_rows(wordlist)%>%
    #tidylo::bind_log_odds(y, term, term_count)
  abhi_bind_log_odds(y, term, term_count)
  
  if('log_odds_weighted' %in% colnames(wordlist)){
    wordlist = wordlist%>%rename(log_odds = log_odds_weighted)
  }
  
  return(wordlist)
}


attach_POS2 <- function(phrase, python.env = 'r-tensorflow', chunksize = 50000){
  spacy_initialize(condaenv = python.env )
  plan(multiprocess)
  number.of.workers = availableCores()
  
  chunksize = min(ceiling(length(phrase)/number.of.workers), chunksize )
  chunk_arr = (1:length(phrase))%/%( length(phrase) / (ceiling(length(phrase)/chunksize)) + 0.01)
  phrase = split(phrase, chunk_arr  ) 
  
  phrase = ( future_lapply(phrase, function(x) {
    txt = spacy_parse(x, entity = F)
    return(txt)
  }))
  plan(sequential)
  
  phrase =  data.table::rbindlist(phrase, use.names = T, idcol = 'listelem')%>%
    mutate(word_pos = str_c(lemma,'_',pos), rn = row_number())%>%
    group_by(listelem, doc_id)%>%
    summarize(postext = paste0(word_pos,collapse=' '), rn = min(rn))%>%
    ungroup()%>%
    arrange(rn)%>%
    pull(postext)
  
  spacy_finalize()
  
  return(phrase)
}

get_embeddings <- function(tokens, ngram.min = 1, ngram.max = 1, ngram.sep="_", apply.tfidf = T, tfidf.norm = 'none'){
  
  it <- itoken(tokens, progressbar = FALSE)
  vocab <- create_vocabulary(it, ngram = c(ngram_min = ngram.min, ngram_max = ngram.max), sep_ngram = ngram.sep)
  vectorizer <- vocab_vectorizer(vocab)

  if (apply.tfidf == T){
    model_tfidf <- TfIdf$new(norm = tfidf.norm)
    train.dtm <- create_dtm(it, vectorizer) %>%
      fit_transform(model_tfidf) 
    colnames(train.dtm) <- vocab$term
  } else {
    train.dtm <- create_dtm(it, vectorizer) 
    colnames(train.dtm) <- vocab$term
    model_tfidf = NA
  }
  
  return(list('vocab' = vocab,
              'model_tfidf' = model_tfidf,
              'word_dtm' = train.dtm))
  
}

get_doc_dtm <-  function(worddtm, wv, svd = F, min = F, max = F){
   common_terms = intersect(colnames(worddtm), rownames(wv) )
  docdtm = worddtm[, common_terms] %*% wv[common_terms, ]
  
  if (svd == T){
    doc.svd = svd(docdtm)
    doc.svd.reconstructed = doc.svd$u[,1,drop=F] %*% diag(doc.svd$d)[1,1, drop=F] %*% t(doc.svd$v[,1,drop=F])
    docdtm = docdtm - doc.svd.reconstructed
    
  }
  
  if (min==T) {
    temp1 = wv[common_terms,] - min(wv[common_terms,]) ### make all numbers positive
    temp11 = worddtm[,common_terms]
    temp11 = sign(temp11)
    docdtm = cbind(  docdtm, ((temp11)^100 %*% temp1^100)^(1/100) )
    rm(temp1)
    rm(temp11)
  }
  
  if (max ==T) {
    temp1 = wv[common_terms,] - min(wv[common_terms,]) ### make all numbers positive
    temp11 = worddtm[,common_terms]
    temp11[temp11==0]=max(temp11)
    temp11 = sign(temp11)
    docdtm = cbind(  docdtm, 1/ ( ((1/temp11)^(100) %*% (1/temp1)^(100))^(1/100 ) ) )
    rm(temp1)
    rm(temp11)
  }
  
  docdtm = as.matrix( docdtm )
  return(docdtm)
  
}

create.rtsne.matrix <- function(fndv, number.of.samples = 10000, rtsne.perplexity = 50, seed = 80) {
  set.seed(seed)
  random.indices = sample(seq(1,nrow(fndv),1), min(nrow(fndv),number.of.samples), replace = F )
  rtsne_out <- Rtsne(as.matrix(fndv[random.indices,]), perplexity = rtsne.perplexity ,check_duplicates = FALSE)
  return(list('rtsne_out'=rtsne_out, 'random.indices' = random.indices) )
  
}

