# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)

#Data import and cleaning
gss_data <- haven::read_sav("../data/GSS2016.sav") #read_sav function not found without library call, though haven is part of the tidyverse
gss_tbl <- tibble(gss_data) %>%
  mutate(workhours = as.integer(HRS1)) %>% #enables logical comparison in next line, no data loss
  filter(workhours != is.na(workhours)) %>% #2867 - (1198 NA + 23 don't know / missing) = 1646 cases as in documentation, Page 123.
  select(which(colMeans(is.na(gss_data)) <= 0.25)) #selecting all columns WHICH have colmeans of missing data less or equal to 25%. #check with colMeans(is.na(gss_tbl)) shows missingness all less than .25