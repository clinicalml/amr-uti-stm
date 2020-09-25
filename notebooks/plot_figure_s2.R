#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(lubridate)
library(gridExtra)

# ROC Curves
roc <- read_csv(file="fig_data/figure_s2.csv")
roc$drug = factor(roc$drug, levels=c("Nitrofurantoin","TMP-SMX","Ciprofloxacin", "Levofloxacin")) 

# ROC plots
roc.plt <- roc %>%
  ggplot(aes(fpr, tpr, color = set)) + 
  geom_line() + 
  facet_wrap(. ~ drug, ncol = 2) +
  theme_minimal() +
  scale_colour_discrete(name="Cohort",
                        breaks=c("full_cohort", "prior_resist_cohort"),
                        labels=c("   Full         ", "   Prior antibiotic\n   resistance or\n   exposure")) +
  theme(legend.position = "bottom",
        legend.background = element_blank(),
        legend.title = element_blank(),
        legend.text = element_text(size = rel(1)),
        panel.border = element_rect(fill = NA, color = "black"),
        plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold", size = rel(1)),
        axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
  guides(colour = guide_legend(override.aes = list(size=1))) + 
  xlab('% false resistance') +
  ylab('% true resistance') + 
  labs(title='ROC curves')
roc.plt

filename <- paste0("figures/figure_s2.pdf")
ggsave(plot = roc.plt, device = "pdf", width = 6, height = 6, units = "in", file = filename)
