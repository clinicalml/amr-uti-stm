#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(lubridate)
library(gridExtra)

#### Figure 3: Thresholds and FNR / FPR Curves ####

# Import data and put in long format
thresh <- read_csv(file="fig_data/figure_3_fpr_fnr.csv")
thresh$drug_new <- factor(thresh$drug, levels=c("Nitrofurantoin","TMP-SMX","Ciprofloxacin", "Levofloxacin")) 

optimal.s <- read_csv(file = "fig_data/figure_3_chosen_thresh.csv") %>%
  mutate(lab = "Susceptible",
         position = 50,
         vline = -0.7,
         ang = 90)

optimal.ns <- optimal.s %>%
  mutate(lab = "Non-susceptible",
         ang = -90) %>%
  filter(!is.na(drug))

optimal <- rbind(optimal.s, optimal.ns) %>% 
  distinct() %>%
  mutate(lab = factor(lab, levels=c("Susceptible", "Non-susceptible")))
optimal$drug_new <- factor(optimal$drug, levels=c("Nitrofurantoin","TMP-SMX","Ciprofloxacin", "Levofloxacin")) 

rm(optimal.s, optimal.ns)

# Plot optimal thresholds with FPR/FNR rate
thresh.plt <- thresh %>% filter(Threshold <= 1) %>%
  ggplot(aes(x = Threshold, y = value)) + 
  geom_line(aes(color = set)) + 
  facet_wrap(. ~ drug_new, ncol = 1) +
  theme_minimal() +
  scale_colour_discrete(breaks=c("FNR", "FPR"),
                        labels=c("  False susceptibility    ", "  False non-susceptibility    ")) +
  geom_text(data = optimal, aes(x = value, y = position, label = lab, angle = ang, vjust = vline), size = 2) +
  geom_vline(data = optimal, aes(xintercept=value), show.legend = F) +
  theme(legend.position = "bottom",
    legend.title = element_blank(),
    legend.background = element_blank(),
    panel.border = element_rect(fill = NA, color = "black"),
    plot.title = element_text(face = "bold", hjust = 0.5),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = rel(0.8)),
    axis.title.y = element_text(size = rel(0.8)),
    axis.text.x = element_text(size = rel(0.8)),
    axis.text.y = element_text(size = rel(0.8)),
    strip.text.x = element_text(face = "bold")) +
  xlab("Probability of non-susceptibility") + ylab("Percent of decisions")

filename <- paste0("figures/figure_3.pdf")
ggsave(plot = thresh.plt, device = "pdf", width = 4, height = 7, units = "in", file = filename)
