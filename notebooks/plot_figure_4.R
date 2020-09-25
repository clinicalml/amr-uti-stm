#!/usr/bin/env Rscript

library(readr)
library(plyr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(lubridate)
library(gridExtra)

#### Figure 4: Post hoc error analysis ####

error <- read_csv(file = "fig_data/figure_4_error_analysis.csv")
error$Drug <- factor(error$Drug, levels=c("Second line","First line")) %>% 
	revalue(c("Second line"="Second line\nagent", "First line"="First line\nagent"))

error$Comparator <- factor(error$Comparator, levels = c("MD_broad_IAT", "MD_broad_noIAT" ,"MD_narrow_IAT" ,"MD_narrow_noIAT")) 

error.plt <- error %>%
  mutate(Result = factor(Result, levels=c("Inappropriate", "Appropriate"))) %>%
  ggplot(aes(x = Drug, y = Value, fill = Result)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.7, 
           position = "dodge", color ="black") +
  coord_flip() + 
  geom_text(aes(label=Value, y = Value+80),
            position = position_dodge(width=.8),
            size=3) +
  scale_fill_brewer(palette="Blues", direction=-1) +
  guides(fill = guide_legend(reverse=T)) + 
  facet_grid(Comparator ~ ., scales = "free_y") +
  theme_minimal() +
  theme(legend.title = element_blank(),
        legend.position = "bottom",
        legend.spacing.x = unit(0.25, 'cm'),
        panel.border = element_rect(fill = NA, color = "black"),
        panel.grid.minor = element_blank(),
        strip.text = element_blank(),
        axis.text.x = element_text(angle = 0, color='black', hjust=1),
        axis.text.y = element_text(angle = 0, color='black', hjust=0.5),
        panel.spacing.x=unit(4, "lines")) + xlab(NULL) + ylab("Frequency")

filename <- paste0("figures/figure_4.pdf")
ggsave(plot = error.plt, device = "pdf", width = 5, height = 5, units = "in", file = filename)

