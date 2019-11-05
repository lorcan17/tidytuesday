---
title: "2019-10-27 NYC Squirrel Census"
output: 
  html_document:
    keep_md: true
editor_options: 
  chunk_output_type: console
---

Read in the data

```r
# Load libraries
library(tidyverse)
```

```
## ── Attaching packages ────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──
```

```
## ✔ ggplot2 3.2.1     ✔ purrr   0.3.2
## ✔ tibble  2.1.3     ✔ dplyr   0.8.3
## ✔ tidyr   1.0.0     ✔ stringr 1.4.0
## ✔ readr   1.3.1     ✔ forcats 0.4.0
```

```
## ── Conflicts ───────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ dplyr::filter() masks stats::filter()
## ✖ dplyr::lag()    masks stats::lag()
```

```r
library(tidymodels)
```

```
## Registered S3 method overwritten by 'xts':
##   method     from
##   as.zoo.xts zoo
```

```
## ── Attaching packages ───────────────────────────────────────────────────────────────────────────────────────── tidymodels 0.0.3 ──
```

```
## ✔ broom     0.5.2     ✔ recipes   0.1.7
## ✔ dials     0.0.3     ✔ rsample   0.0.5
## ✔ infer     0.5.0     ✔ yardstick 0.0.4
## ✔ parsnip   0.0.4
```

```
## ── Conflicts ──────────────────────────────────────────────────────────────────────────────────────────── tidymodels_conflicts() ──
## ✖ scales::discard() masks purrr::discard()
## ✖ dplyr::filter()   masks stats::filter()
## ✖ recipes::fixed()  masks stringr::fixed()
## ✖ dplyr::lag()      masks stats::lag()
## ✖ dials::margin()   masks ggplot2::margin()
## ✖ dials::offset()   masks stats::offset()
## ✖ yardstick::spec() masks readr::spec()
## ✖ recipes::step()   masks stats::step()
```

```r
library(lubridate)
```

```
## 
## Attaching package: 'lubridate'
```

```
## The following object is masked from 'package:base':
## 
##     date
```

```r
library(forcats)
library(knitr)
library(janitor)
```

```
## 
## Attaching package: 'janitor'
```

```
## The following objects are masked from 'package:stats':
## 
##     chisq.test, fisher.test
```

```r
library(xgboost)
```

```
## 
## Attaching package: 'xgboost'
```

```
## The following object is masked from 'package:dplyr':
## 
##     slice
```



```r
nyc_squirrels_raw <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-10-29/nyc_squirrels.csv")
```

```
## Parsed with column specification:
## cols(
##   .default = col_character(),
##   long = col_double(),
##   lat = col_double(),
##   date = col_double(),
##   hectare_squirrel_number = col_double(),
##   running = col_logical(),
##   chasing = col_logical(),
##   climbing = col_logical(),
##   eating = col_logical(),
##   foraging = col_logical(),
##   kuks = col_logical(),
##   quaas = col_logical(),
##   moans = col_logical(),
##   tail_flags = col_logical(),
##   tail_twitches = col_logical(),
##   approaches = col_logical(),
##   indifferent = col_logical(),
##   runs_from = col_logical(),
##   zip_codes = col_double(),
##   community_districts = col_double(),
##   borough_boundaries = col_double()
##   # ... with 2 more columns
## )
```

```
## See spec(...) for full column specifications.
```

```r
nyc_squirrels <- nyc_squirrels_raw %>% mutate(shift = as_factor(shift),
                                              date = mdy(date),
                                              age = fct_explicit_na(fct_relevel(age,"Juvenile")),
                                              primary_fur_color = fct_explicit_na(primary_fur_color),
                                              #highlight_fur_color = fct_explicit_na(highlight_fur_color))
                                              location = fct_explicit_na(fct_relevel(location,"Ground Plane")))

nyc_squirrels <- nyc_squirrels %>%  select(date,age,primary_fur_color,location,running,chasing,climbing,eating,foraging, kuks,tail_flags,approaches,indifferent,runs_from)
```

This is a great dataset with lots of variables. I'm going to use this dataset to practise my machine learning techniques.

I'm going to create a model that can accuractely predict if a squirrel is on the ground or above the ground.

I'm going to be using the tidymodels workflow, and following this helpful guide:
https://towardsdatascience.com/modelling-with-tidymodels-and-parsnip-bae2c01c131c
https://www.benjaminsorensen.me/post/modeling-with-parsnip-and-tidymodels/
https://www.alexpghayes.com/blog/implementing-the-super-learner-with-tidymodels/

Given that there are only 64 missing data points for location, I will remove these.

```r
nyc_squirrels %>% count(location)
```

```
## # A tibble: 3 x 2
##   location         n
##   <fct>        <int>
## 1 Ground Plane  2116
## 2 Above Ground   843
## 3 (Missing)       64
```

```r
nyc_squirrels <- nyc_squirrels %>% filter(location != "(Missing)") %>% droplevels()

levels(nyc_squirrels$location)
```

```
## [1] "Ground Plane" "Above Ground"
```

### Train and Test Split

Firstly, create a randomised training and test plit of the original data.

```r
set.seed(seed = 1) 
train_test_split <- initial_split(data = nyc_squirrels,prop = 0.8)

train_set <- training(train_test_split)
test_set <- testing(train_test_split)
```

### Create recipe
Here we prep a recipe for how we want to transform our data and then apply these steps to the train and test set

```r
set.seed(seed = 1) 
recipe <- recipe(location ~ ., data = train_set) %>% 
  step_dummy(primary_fur_color,age,one_hot = T) %>% 
  prep(train_set)

train_baked <- bake(recipe,new_data = train_set)
test_baked <- bake(recipe,new_data = test_set)
```

### Fit the model

```r
set.seed(seed = 1) 
boost_tree <- boost_tree(mode = "classification") %>% 
  set_engine("xgboost") %>% 
  fit(location ~ ., data = train_baked)

importance <- xgb.importance(feature_names = colnames(train_baked %>% select(-location)),model = boost_tree$fit)

xgb.plot.importance(importance)
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

### Performance Assesment

```r
predictions_xgb <- boost_tree %>% 
  predict(new_data = test_baked) %>% 
  bind_cols(test_baked %>% select(location))
```

There are several metrics that can be used to investigate the performance of a classification model but for simplicity I’m only focusing on a selection of them: accuracy, precision, recall and F1_Score.


```r
predictions_xgb %>%
  conf_mat(location, .pred_class) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-6-1.png)<!-- -->


```r
predictions_xgb %>%
  conf_mat(location, .pred_class) %>% summary() %>% 
  select(-.estimator) %>%
  filter(.metric %in%
    c("accuracy", "precision", "recall", "f_meas")) %>%
  kable(digits = 2)
```



.metric      .estimate
----------  ----------
accuracy          0.84
precision         0.84
recall            0.95
f_meas            0.89

Wow, so on first go it seems we have built a model which can fairly accuractely predict if a squirrel is above ground or not.


```r
boost_tree %>% 
  predict(new_data = test_baked,type = "prob") %>% clean_names %>% select(pred_ground_plane) %>% 
  bind_cols(test_baked %>% select(location)) %>% 
  gain_curve(truth = location,pred_ground_plane) %>% autoplot()
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

### Hyper Parameter Tuning
Create a random grid of parameters object values. Parameters we can tune can be found here: 
https://tidymodels.github.io/parsnip/reference/boost_tree.html

I have chosen to tune mtry, learn rate, and tree depth only but could have tuned all found in the above link.


```r
library(dials)
set.seed(seed = 1) 

xgb_grid <- grid_random(
mtry(c(1,10)),
learn_rate(c(0,1),trans = NULL),
tree_depth(c(1,20)),
size = 10) %>% mutate(model_id = row_number()) 

xgb_grid
```

```
## # A tibble: 10 x 4
##     mtry learn_rate tree_depth model_id
##  * <int>      <dbl>      <int>    <int>
##  1     9      0.770          9        1
##  2     4      0.498         14        2
##  3     7      0.718          5        3
##  4     1      0.992          5        4
##  5     2      0.380          2        5
##  6     7      0.777         10        6
##  7     2      0.935         12        7
##  8     3      0.212         15        8
##  9     1      0.652          1        9
## 10     5      0.126         20       10
```

Now use K-Fold Cross-Validation to determine which set of parameters is the most accurate. 

Reminder:
3-Fold CV is split into 3 equal sized groups and is tested like below:
*Model1: Trained on Fold1 + Fold2, Tested on Fold3
*Model2: Trained on Fold2 + Fold3, Tested on Fold1
*Model3: Trained on Fold1 + Fold3, Tested on Fold2

https://machinelearningmastery.com/k-fold-cross-validation/

### Tune 1


```r
set.seed(seed = 1) 
folds <- vfold_cv(train_set,6,2)

set.seed(seed = 1) 
folded <-  folds %>% 
  # Cross join the xvg grid create above
  expand_grid(xgb_grid) %>% 
  # Then create analysis and assesment sets
  mutate(analysis = map(splits, analysis),
  assesment = map(splits, assessment),
  # Create a Recipe. Prepper is a wrapper for `prep()` which handles `split` objects
  recipe = map(splits,~prepper(.x,recipe(location ~ ., data = train_set) %>% 
  step_dummy(primary_fur_color,age,one_hot = T))),
  # Now bake analysis and assesment sets
  analysis = map2(recipe,analysis,bake),
  assesment = map2(recipe,assesment,bake),
  # Run model against analysis set with values for parameter objects defined
  boost_tree = pmap(list(mtry,learn_rate,tree_depth,analysis),
  ~ boost_tree(mode = "classification",mtry= ..1,learn_rate = ..2,tree_depth= ..3) %>%
  set_engine("xgboost") %>%
  fit(location ~ ., data = ..4)
  ),
  # Create predictions for each model on the assesment set
  predictions_xgb = map2(
  assesment,
  boost_tree,
  ~
  predict(.y, new_data = .x) %>%
  bind_cols(.x %>% select(location))
  ),
  # Compute accuracy metric
  accuracy = map(predictions_xgb,~ accuracy(., truth = location, estimate = .pred_class))
  ) %>% 
  unnest(accuracy)
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```r
# Visualise model performance by model id
folded %>% ggplot(aes(x = as_factor(model_id), y = .estimate)) + geom_boxplot()
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

```r
# Table of top model combinations
hi <- folded %>% group_by(model_id) %>% summarise(.estimate = median(.estimate)) %>% ungroup( )%>% filter(.estimate >0.85) %>% 
  inner_join(xgb_grid) %>% arrange(desc(.estimate))
```

```
## Joining, by = "model_id"
```

```r
hi
```

```
## # A tibble: 1 x 5
##   model_id .estimate  mtry learn_rate tree_depth
##      <int>     <dbl> <int>      <dbl>      <int>
## 1        9     0.853     1      0.652          1
```

```r
#  Variance of model accuracy by paramter values
folded %>% select(mtry:model_id,.estimate) %>% group_by(model_id)%>% pivot_longer(mtry:tree_depth) %>% ggplot(aes(x = value,y = .estimate,colour = as_factor(model_id))) + geom_point() + facet_wrap(~name,scales ="free_x") 
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-10-2.png)<!-- -->

```r
# Mean accuracy for each parameter values
folded %>% select(mtry:model_id,.estimate) %>% group_by(model_id)%>% pivot_longer(mtry:tree_depth) %>% group_by(name,value) %>% summarise(.estimate = mean(.estimate)) %>% ggplot(aes(x = value,y = .estimate)) + geom_point() + facet_wrap(~name,scales ="free_x") 
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-10-3.png)<!-- -->

Looking at the learn_rate graph we see that learn rates < 0.5 perform worse. Let's edit this parameter to only include 0.5 to 1 and rerun. Tree depth also looks like it can be reduced to between 1 and 10 but I will only make one change at a time.

# Tune 2

```r
set.seed(seed = 2) 
xgb_grid <- grid_random(
mtry(c(1,10)),
learn_rate(c(0.5,1),trans = NULL),
tree_depth(c(1,20)),
size = 10) %>% mutate(model_id = row_number()) 

xgb_grid
```

```
## # A tibble: 10 x 4
##     mtry learn_rate tree_depth model_id
##  * <int>      <dbl>      <int>    <int>
##  1     5      0.988          9        1
##  2     6      0.613         16        2
##  3     6      0.722          4        3
##  4     8      0.537         11        4
##  5     1      0.831          6        5
##  6     1      0.694          9        6
##  7     9      0.918         14        7
##  8     2      0.575          8        8
##  9     1      0.674         16        9
## 10     3      0.744         13       10
```


```r
set.seed(seed = 1) 
folds <- vfold_cv(train_set,6,2)

set.seed(seed = 1) 
folded <-  folds %>% 
  # Cross join the xvg grid create above
  expand_grid(xgb_grid) %>% 
  # Then create analysis and assesment sets
  mutate(analysis = map(splits, analysis),
  assesment = map(splits, assessment),
  # Create a Recipe. Prepper is a wrapper for `prep()` which handles `split` objects
  recipe = map(splits,~prepper(.x,recipe(location ~ ., data = train_set) %>% 
  step_dummy(primary_fur_color,age,one_hot = T))),
  # Now bake analysis and assesment sets
  analysis = map2(recipe,analysis,bake),
  assesment = map2(recipe,assesment,bake),
  # Run model against analysis set with values for parameter objects defined
  boost_tree = pmap(list(mtry,learn_rate,tree_depth,analysis),
  ~ boost_tree(mode = "classification",mtry= ..1,learn_rate = ..2,tree_depth= ..3) %>%
  set_engine("xgboost") %>%
  fit(location ~ ., data = ..4)
  ),
  # Create predictions for each model on the assesment set
  predictions_xgb = map2(
  assesment,
  boost_tree,
  ~
  predict(.y, new_data = .x) %>%
  bind_cols(.x %>% select(location))
  ),
  # Compute accuracy metric
  accuracy = map(predictions_xgb,~ accuracy(., truth = location, estimate = .pred_class))
  ) %>% 
  unnest(accuracy)
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```r
# Visualise model performance by model id
folded %>% ggplot(aes(x = as_factor(model_id), y = .estimate)) + geom_boxplot()
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

```r
# Table of top model combinations
hi <- folded %>% group_by(model_id) %>% summarise(.estimate = median(.estimate)) %>% ungroup( )%>% filter(.estimate >0.85) %>% 
  inner_join(xgb_grid) %>% arrange(desc(.estimate))
```

```
## Joining, by = "model_id"
```

```r
hi
```

```
## # A tibble: 4 x 5
##   model_id .estimate  mtry learn_rate tree_depth
##      <int>     <dbl> <int>      <dbl>      <int>
## 1        3     0.853     6      0.722          4
## 2       10     0.853     3      0.744         13
## 3        4     0.852     8      0.537         11
## 4        5     0.850     1      0.831          6
```

```r
#  Variance of model accuracy by paramter values
folded %>% select(mtry:model_id,.estimate) %>% group_by(model_id)%>% pivot_longer(mtry:tree_depth) %>% ggplot(aes(x = value,y = .estimate,colour = as_factor(model_id))) + geom_point() + facet_wrap(~name,scales ="free_x") 
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-12-2.png)<!-- -->

```r
# Mean accuracy for each parameter values
folded %>% select(mtry:model_id,.estimate) %>% group_by(model_id)%>% pivot_longer(mtry:tree_depth) %>% group_by(name,value) %>% summarise(.estimate = mean(.estimate)) %>% ggplot(aes(x = value,y = .estimate)) + geom_point() + facet_wrap(~name,scales ="free_x") 
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-12-3.png)<!-- -->

From the above it apears an mtry around 5 works best. Let's change that parameter next.

### Tune 3


```r
set.seed(seed = 3) 
xgb_grid <- grid_random(
mtry(c(7,7)),
learn_rate(c(0.5,0.8),trans = NULL),
tree_depth(c(1,5)),
size = 10) %>% mutate(model_id = row_number()) 

xgb_grid
```

```
## # A tibble: 10 x 4
##     mtry learn_rate tree_depth model_id
##  * <int>      <dbl>      <int>    <int>
##  1     5      0.660          5        1
##  2     2      0.667          5        2
##  3     4      0.760          1        3
##  4     7      0.749          1        4
##  5     4      0.533          2        5
##  6     2      0.711          1        6
##  7     3      0.769          4        7
##  8     7      0.584          4        8
##  9     4      0.568          3        9
## 10     2      0.505          2       10
```


```r
set.seed(seed = 1) 
folds <- vfold_cv(train_set,6,2)

set.seed(seed = 1) 
folded <-  folds %>% 
  # Cross join the xvg grid create above
  expand_grid(xgb_grid) %>% 
  # Then create analysis and assesment sets
  mutate(analysis = map(splits, analysis),
  assesment = map(splits, assessment),
  # Create a Recipe. Prepper is a wrapper for `prep()` which handles `split` objects
  recipe = map(splits,~prepper(.x,recipe(location ~ ., data = train_set) %>% 
  step_dummy(primary_fur_color,age,one_hot = T))),
  # Now bake analysis and assesment sets
  analysis = map2(recipe,analysis,bake),
  assesment = map2(recipe,assesment,bake),
  # Run model against analysis set with values for parameter objects defined
  boost_tree = pmap(list(mtry,learn_rate,tree_depth,analysis),
  ~ boost_tree(mode = "classification",mtry= ..1,learn_rate = ..2,tree_depth= ..3) %>%
  set_engine("xgboost") %>%
  fit(location ~ ., data = ..4)
  ),
  # Create predictions for each model on the assesment set
  predictions_xgb = map2(
  assesment,
  boost_tree,
  ~
  predict(.y, new_data = .x) %>%
  bind_cols(.x %>% select(location))
  ),
  # Compute accuracy metric
  accuracy = map(predictions_xgb,~ accuracy(., truth = location, estimate = .pred_class))
  ) %>% 
  unnest(accuracy)
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..1 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..3 may be used in an incorrect context
```

```
## Warning: <anonymous>: ..2 may be used in an incorrect context
```

```r
# Visualise model performance by model id
folded %>% ggplot(aes(x = as_factor(model_id), y = .estimate)) + geom_boxplot()
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

```r
# Table of top model combinations
hi <- folded %>% group_by(model_id) %>% summarise(.estimate = median(.estimate)) %>% ungroup( )%>% filter(.estimate >0.85) %>% 
  inner_join(xgb_grid) %>% arrange(desc(.estimate))
```

```
## Joining, by = "model_id"
```

```r
hi
```

```
## # A tibble: 3 x 5
##   model_id .estimate  mtry learn_rate tree_depth
##      <int>     <dbl> <int>      <dbl>      <int>
## 1        4     0.853     7      0.749          1
## 2        3     0.852     4      0.760          1
## 3        8     0.850     7      0.584          4
```

```r
#  Variance of model accuracy by paramter values
folded %>% select(mtry:model_id,.estimate) %>% group_by(model_id)%>% pivot_longer(mtry:tree_depth) %>% ggplot(aes(x = value,y = .estimate,colour = as_factor(model_id))) + geom_point() + facet_wrap(~name,scales ="free_x") 
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-14-2.png)<!-- -->

```r
# Mean accuracy for each parameter values
folded %>% select(mtry:model_id,.estimate) %>% group_by(model_id)%>% pivot_longer(mtry:tree_depth) %>% group_by(name,value) %>% summarise(.estimate = mean(.estimate)) %>% ggplot(aes(x = value,y = .estimate)) + geom_point() + facet_wrap(~name,scales ="free_x") 
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-14-3.png)<!-- -->

## Choose parameters
Our best performing model was using the following parameters:

Now let's use this on the testing set.

### Fit the model

```r
set.seed(seed = 1) 
boost_tree <- boost_tree(mode = "classification",mtry= 7,learn_rate = 0.75,tree_depth= 1) %>% 
  set_engine("xgboost") %>% 
  fit(location ~ ., data = train_baked)

importance <- xgb.importance(feature_names = colnames(train_baked %>% select(-location)),model = boost_tree$fit)

xgb.plot.importance(importance)
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

### Performance Assesment

```r
predictions_xgb <- boost_tree %>% 
  predict(new_data = test_baked) %>% 
  bind_cols(test_baked %>% select(location))
```

There are several metrics that can be used to investigate the performance of a classification model but for simplicity I’m only focusing on a selection of them: accuracy, precision, recall and F1_Score.


```r
predictions_xgb %>%
  conf_mat(location, .pred_class) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes(Prediction, Truth, alpha = n)) +
  geom_tile(show.legend = FALSE) +
  geom_text(aes(label = n), colour = "white", alpha = 1, size = 8)
```

![](2019-10-27-NYC-Squirrel-Cencus_files/figure-html/unnamed-chunk-17-1.png)<!-- -->


```r
predictions_xgb %>%
  conf_mat(location, .pred_class) %>% summary() %>% 
  select(-.estimator) %>%
  filter(.metric %in%
    c("accuracy", "precision", "recall", "f_meas")) %>%
  kable(digits = 2)
```



.metric      .estimate
----------  ----------
accuracy          0.83
precision         0.83
recall            0.94
f_meas            0.89

All in all, not a bad score. Although I wonder if we could have gotten similar score by the following logic:
If Swquirrel Climbing then Above Plane


```r
nyc_squirrels %>% 
  mutate(guess = as_factor(if_else(climbing,'Above Ground','Ground Plane'))) %>% 
  accuracy(estimate = guess,truth = location)
```

```
## # A tibble: 1 x 3
##   .metric  .estimator .estimate
##   <chr>    <chr>          <dbl>
## 1 accuracy binary         0.845
```

Well, it turns out guessing it's Above Plane if it's climbing gets you a accuracy of %84.5 which is more than our model!

Back to the drawing board...

### Next Steps: Fit the metalearner
Use a multinomial regression as the metalearner.



