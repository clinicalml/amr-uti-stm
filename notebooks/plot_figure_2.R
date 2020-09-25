#!/usr/bin/env Rscript

library(readr)
library(dplyr)
library(tibble)
library(ggplot2)
library(cowplot)
library(lubridate)
library(gridExtra)

#### Figure 2: IAT vs. 2nd line usage Plot ####

iat <- read_csv(file = "fig_data/figure_2_threshold_sensitivity.csv") %>%
  mutate(spectrum = broad_prop*100,
         iat = iat_prop*100)

thresh.choice <- data.frame(choice = "Final", 
                            spectrum = c(9.6), 
                            iat = c(10.18),
                            group = c("iat"))

iat.plt <- iat %>%
  ggplot(aes(x = spectrum, y = iat)) + 
  geom_point(shape=19, alpha=1/4) +
  geom_point(data = filter(thresh.choice, choice == "Final" & group == "iat"), color = "blue", size = 3) +
  theme_minimal() +
  theme(legend.position = "none",
        panel.border = element_rect(fill = NA, color = "black"),
        plot.title = element_text(hjust = 0.5),
        axis.title.x = element_text(size = rel(0.8)),
        axis.title.y = element_text(size = rel(0.8)),
        axis.text.x = element_text(size = rel(0.8)),
        axis.text.y = element_text(size = rel(0.8)),
        panel.grid.minor = element_blank()) + xlab("Percent usage of ciprofloxacin or levofloxacin") + ylab("Percent IAT")

filename <- paste0("figures/figure_2.pdf")
ggsave(plot = iat.plt, device = "pdf", width = 8, height = 7, units = "in", file = filename)
