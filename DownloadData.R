
## this is the R script to download voice files from synapse (www.synapse.org)
## you can modify it into python script following synapse wiki pages
## e.g. http://python-docs.synapse.org or http://r-docs.synapse.org

# connect to synapse and login
require(synapseClient)
synapseLogin() # upon prompt, type your id and password

# use one of the following query to download files

# just download audio files
tq <- synTableQuery('SELECT audio_audiom4a FROM syn4598512) 

# or download the whole table
# tq <- synTableQuery('SELECT * FROM syn4598512) 

# or download limited number of audio files but started at a particular row
# tq <- synTableQuery('SELECT audio_audiom4a FROM syn4598512 limit 2662 offset 13087')

# after you set up the query, here is how you download the data
sc <- synapseClient:::synGetColumns(tq@schema)
theseCols <- sapply(as.list(1:length(sc)), function(x){
  if(sc[[x]]@columnType=="FILEHANDLEID"){
    return(sc[[x]]@name)
  } else{
    return(NULL)
  }
})
theseCols <- unlist(theseCols)

theseFiles <- lapply(as.list(theseCols), function(cc){
  sapply(as.list(rownames(tq@values)), function(rn){
    synDownloadTableFile(tq, rn, cc)
  })
})
names(theseFiles) <- theseCols
