compdata <- read.csv("C:/Users/timet/Documents/projectCSUN/csunThesisRepo/data/compexp/compexpData.csv")


export <- compdata %>%
  filter(species == "rotifer") %>%
  mutate(uniqueID = as.factor(paste(repNum, repID, sep = "")),
         day = as.integer(day)) %>%
  select(day, uniqueID, count)

