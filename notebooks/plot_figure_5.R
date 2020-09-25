#!/usr/bin/env Rscript

library(readr)
library(plyr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(lubridate)
library(gridExtra)

#### Figure 5: Feature Importance ####

ablation <- read_csv(file = "fig_data/figure_5.csv") %>%
  mutate(drug = factor(drug, levels = c("Nitrofurantoin", "TMP-SMX", "Ciprofloxacin", "Levofloxacin")),
         Model = factor(model, levels = c("Full",  "Prior antibiotics", "Prior resistance",
                                          "Colonization pressure", "Hospital antibiotic use"))) %>%
  filter(cohort == 'full')

# Plot absolute AUROCs
abl.plt <- ablation %>%
  filter(!Model %in% c("Full")) %>%
  ggplot(aes(x = Model, y = AUC)) +
  geom_point(size = 3) +
  facet_grid(. ~ drug) +
  geom_errorbar(aes(x = Model, ymin = LCI, ymax = UCI), width=0.4, alpha=0.9) +
  geom_hline(data = filter(ablation, Model == "Full"), aes(yintercept = AUC), color = "red", linetype = "dotted", show.legend = F) +
  geom_hline(data = filter(ablation, Model == "Full"), aes(yintercept = LCI), color = "dark blue", linetype = "dotted", show.legend = F) +
  geom_hline(data = filter(ablation, Model == "Full"), aes(yintercept = UCI), color = "dark blue", linetype = "dotted", show.legend = F) +
  theme_minimal() + 
  theme(legend.title = element_text(),
        legend.position = "none",
        panel.border = element_rect(fill = NA, color = "black", size = rel(1)),
        plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, color='black'),
        axis.title.x = element_text(face='bold'),
        strip.text.x = element_text(face = "bold")) +
  #scale_y_continuous(limits = c(0.5, 0.7)) +
  xlab('\nHeld-out feature set') + ylab("AUROC")

filename <- paste0("figures/figure_5.pdf")
ggsave(plot = abl.plt, device = "pdf", width = 8, height = 7, units = "in", file = filename)
