library(topicmodels)
#data()
library(tm)
library(wordcloud)
library(SnowballC)
library(fpc)
library(RTextTools)
library(tm)

data("NYTimes")

text <- NYTimes

head(text)
head(text['Title'])
head(text$Subject)

txt.process2 <- function(text) {
  action.corpus <- Corpus(VectorSource(text))
  
  # to lower case
  action.corpus <- tm_map(action.corpus, content_transformer(tolower))
  
  # remove punctuation
  action.corpus <- tm_map(action.corpus, removePunctuation)
  # remove numbers
  # action.corpus <- tm_map(action.corpus, removeNumbers)
  
  # remove URL
  removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
  action.corpus <- tm_map(action.corpus, removeURL)  
  # remove stopwords
  action.corpus <- tm_map(action.corpus, removeWords, stopwords("english"))  
  action.corpus <- tm_map(action.corpus,  removeNumbers)
  #action.corpus.bak <- action.corpus
  
  # remove extra spaces
  action.corpus <- tm_map(action.corpus, stripWhitespace)
  
  action.corpus <- tm_map(action.corpus, stemDocument, language="english")
  #inspect(action.corpus)
  
  # Use PlainTextDocument in tm >= 0.6
  action.corpus <- tm_map(action.corpus, PlainTextDocument)
  # stem completion
  #action.corpus <- tm_map(action.corpus, stemCompletion, dictionary=action.corpus.bak)
  action.tdm <- TermDocumentMatrix(action.corpus, control = list(wordLengths=c(3, Inf)))
  action.dtm <- as.DocumentTermMatrix(action.tdm)
  action.tdm2 <- removeSparseTerms(action.tdm, sparse = 0.95)
  
  list(corpus=action.corpus, dtm<-action.dtm, tdm=action.tdm, tdm2=action.tdm2)
}

out <- txt.process2(text$Subject)
#out$tdm


# k is related with number of documents
lda_model <- LDA(out$tdm, 20, method = 'Gibbs')
lda_inf <- posterior(lda_model, out$tdm)


lda_model2 <- LDA(out$tdm2, 20)
lda_inf2 <- posterior(lda_model2, out$tdm2)


