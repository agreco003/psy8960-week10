# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(caret)

#Data import and cleaning
gss_data <- haven::read_sav("../data/GSS2016.sav") #read_sav function not found without library call, though haven is part of the tidyverse
gss_tbl <- tibble(gss_data) %>%
  #mutate(workhours = HRS1) %>%
  #filter()