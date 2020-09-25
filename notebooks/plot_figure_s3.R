#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(lubridate)
library(gridExtra)

#### Figure S3: Calibration Curves ####

calibration <- read_csv(file = "fig_data/figure_s3.csv") %>%
  mutate(drug = factor(drug, levels = c("Nitrofurantoin", "TMP-SMX", "Ciprofloxacin", "Levofloxacin")))

calib.curve.plt <- calibration %>%
  ggplot(aes(predicted, prop_positive, color = drug, label = count)) +
  geom_point(aes(size = count)) + geom_line() + geom_text(nudge_y = 0.1) + 
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", alpha = 0.5) + 
  scale_size_continuous(range = c(0, 5)) +
  theme_minimal() +
  facet_grid(drug ~ .) +
  theme(legend.position = "none",
        panel.border = element_rect(fill = NA, color = "black"),
        plot.title = element_text(hjust = 0.5),
        panel.grid.minor = element_blank(),
        strip.text.y = element_text(face = "bold")) +
  scale_x_continuous(breaks = c(seq(0, 1, 0.1))) +
  scale_y_continuous(breaks = c(seq(0, 1, 0.25)), limits = c(0,1.1)) +
  labs(title='Calibration plots') + xlab('Mean predicted value') + ylab('Fraction of positives')

filename <- paste0("figures/figure_s3.pdf")
ggsave(plot = calib.curve.plt, device = "pdf", width = 8, height = 6, units = "in", file = filename)
