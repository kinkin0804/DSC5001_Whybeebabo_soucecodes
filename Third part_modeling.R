suppressPackageStartupMessages({
  library(tidymodels); library(glmnet); library(kknn)
  library(readr); library(dplyr); library(tidyr); library(stringr); library(purrr)
})

SAMPLE_MODE  <- "main"   
GRID_LEVELS  <- 10      
SEED         <- 20261116

if (requireNamespace("rstudioapi", quietly = TRUE) &&
    tryCatch(rstudioapi::isAvailable(), error = function(e) FALSE)) {
  setwd(dirname(rstudioapi::getSourceEditorContext()$path))
}
set.seed(SEED)

OUT <- file.path("output", SAMPLE_MODE)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT, "figures"), recursive = TRUE, showWarnings = FALSE)

dat <- read_csv("data/apartment_rent_preprocessed.csv", show_col_types = FALSE) %>%
  mutate(cityname = factor(cityname), state = factor(state))

cat("成员一数据行数：", nrow(dat), "\n")
n0 <- nrow(dat)

flag <- function(x) as.logical(as.integer(x))
if (SAMPLE_MODE == "main") {
  dat <- dat %>% filter(!flag(is_outlier_price),
                        !flag(is_outlier_square_feet),
                        flag(is_geo_valid),
                        !flag(is_business_error),
                        !flag(is_dup_listing))
}
cat("剔除后建模样本：", nrow(dat), "（剔除", n0 - nrow(dat), "行）\n")

AMENITIES <- c("Parking","Dishwasher","Pool","Refrigerator","Patio/Deck","Cable or Satellite",
               "Storage","Gym","Internet Access","Clubhouse","Garbage Disposal","Washer Dryer",
               "Fireplace","Playground","AC","Elevator","Tennis","Gated","Wood Floors","Hot Tub",
               "Basketball","TV","View","Doorman","Alarm","Golf","Luxury")

amen_raw     <- dat$amenities
amen_unknown <- is.na(amen_raw) |
                str_trim(tolower(as.character(amen_raw))) %in% c("", "none", "null", "nan")

for (a in AMENITIES) {
  dat[[paste0("amen_", make.names(a))]] <-
    as.integer(!amen_unknown & str_detect(as.character(amen_raw), fixed(a)))
}
dat$amen_unknown <- as.integer(amen_unknown)   
amenity_cols <- c(paste0("amen_", make.names(AMENITIES)), "amen_unknown")

cat("有设施记录的条数：", sum(!amen_unknown), " 设施未知：", sum(amen_unknown), "\n")
cat("set4 新增设施列数：", length(amenity_cols), "\n")

feature_set <- list(
  set1 = c("square_feet"),
  set2 = c("square_feet","bedrooms","bathrooms"),
  set3 = c("square_feet","bedrooms","bathrooms","cityname","state"),
  set4 = c("square_feet","bedrooms","bathrooms","cityname","state", amenity_cols)
)

prop_split <- initial_split(dat %>% distinct(property_key), prop = 0.8)
prop_train <- training(prop_split); prop_test <- testing(prop_split)

train_data <- dat %>% filter(property_key %in% prop_train$property_key) %>%
  mutate(fix_glmnet = 0)   
test_data  <- dat %>% filter(property_key %in% prop_test$property_key) %>%
  mutate(fix_glmnet = 0)

stopifnot(length(intersect(train_data$property_key, test_data$property_key)) == 0)

cat("训练集：", nrow(train_data), " 测试集：", nrow(test_data),
    " 训练房源：", n_distinct(train_data$property_key),
    " 测试房源：", n_distinct(test_data$property_key), "\n")

cv_folds <- group_vfold_cv(train_data, group = property_key, v = 5)
cat("CV：5 折，按 property_key 分组\n")

lm_spec    <- linear_reg() %>% set_engine("lm") %>% set_mode("regression")
ridge_spec <- linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet") %>% set_mode("regression")
lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>% set_mode("regression")
pen_grid   <- grid_regular(penalty(range = c(-6, 1)), levels = GRID_LEVELS)

knn_spec <- nearest_neighbor(neighbors = tune(), weight_func = "inv", dist_power = 2) %>%
  set_engine("kknn") %>% set_mode("regression")
knn_grid <- grid_regular(neighbors(range = c(5, 100)), levels = GRID_LEVELS)

model_list <- list(
  linear = list(spec = lm_spec,    grid = NULL),
  ridge  = list(spec = ridge_spec, grid = pen_grid),
  lasso  = list(spec = lasso_spec, grid = pen_grid),
  knn    = list(spec = knn_spec,   grid = knn_grid)
)

make_rec <- function(fset_name, f_cols, fml) {
  rec <- recipe(fml, data = train_data)
  if (fset_name %in% c("set3","set4")) {         
    rec <- rec %>% step_novel(all_nominal_predictors()) %>%   
      step_unknown(all_nominal_predictors())                  
  }
  rec <- rec %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors(), -any_of("fix_glmnet"))  
  num_cont <- intersect(c("square_feet","bedrooms","bathrooms"), f_cols)
  if (length(num_cont) > 0) rec <- step_normalize(rec, all_of(num_cont))  
  rec
}

cv_all <- list(); best_all <- list(); pred_all <- list()
t_start <- Sys.time()

for (fset_name in names(feature_set)) {
  f_cols <- feature_set[[fset_name]]
  f_cols2 <- if (fset_name == "set1") c(f_cols, "fix_glmnet") else f_cols
  fml <- as.formula(str_c("price ~ ", str_c(f_cols2, collapse = " + ")))
  cat("\n============ ", fset_name, "：", deparse1(fml), "\n")

  for (m_name in names(model_list)) {
    m  <- model_list[[m_name]]
    wf <- workflow() %>% add_recipe(make_rec(fset_name, f_cols, fml)) %>% add_model(m$spec)
    t0 <- Sys.time()

    if (is.null(m$grid)) {
      res <- fit_resamples(wf, resamples = cv_folds,
                           metrics = metric_set(rmse, mae, rsq),
                           control = control_resamples(save_pred = FALSE))
      met  <- collect_metrics(res) %>% mutate(model = m_name, feature_set = fset_name,
                                             penalty = NA_real_, neighbors = NA_real_)
      best <- tibble(model = m_name, feature_set = fset_name, penalty = NA_real_,
                     neighbors = NA_real_, .config = NA_character_)
      final_fit <- fit(wf, data = train_data)
    } else {
      res <- tune_grid(wf, resamples = cv_folds, grid = m$grid,
                       metrics = metric_set(rmse, mae, rsq),
                       control = control_grid(save_pred = FALSE, allow_par = FALSE))
      met  <- collect_metrics(res) %>% mutate(model = m_name, feature_set = fset_name)
      best <- select_best(res, metric = "rmse")
      final_fit <- fit(finalize_workflow(wf, best), data = train_data)
    }

    cv_all[[paste(fset_name, m_name)]] <- met
    param_txt <- if (is.null(m$grid)) "无调参" else {
      parts <- c(if ("penalty"   %in% names(best)) paste0("penalty=",   signif(best$penalty, 4)),
                 if ("neighbors" %in% names(best)) paste0("neighbors=", best$neighbors))
      paste(parts, collapse = ", ")
    }
    cat("  最优参数：", param_txt, " 用时 ",
        round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), "min\n")

    key <- if (is.null(m$grid))
      met %>% filter(.metric == "rmse") %>% select(.config)
    else best %>% select(.config)
    for (col in c("penalty", "neighbors")) if (!col %in% names(key)) key[[col]] <- NA_real_
    if ("penalty"   %in% names(best)) key$penalty   <- best$penalty
    if ("neighbors" %in% names(best)) key$neighbors <- best$neighbors
    key <- key %>% select(.config, penalty, neighbors)

    met_k <- met %>% semi_join(key, by = ".config") %>%
      select(.metric, .estimate = mean, .config) %>%
      pivot_wider(names_from = .metric, values_from = .estimate)

    best_all[[paste(fset_name, m_name)]] <- tibble(
      model = m_name, feature_set = fset_name,
      penalty = key$penalty, neighbors = key$neighbors, .config = key$.config,
      cv_rmse = met_k$rmse, cv_mae = met_k$mae, cv_rsq = met_k$rsq,
      n_train = nrow(train_data), n_test = nrow(test_data))

    pred_all[[paste(fset_name, m_name)]] <- tibble(
      listing_id = test_data$id, y_true = test_data$price,
      y_pred = predict(final_fit, new_data = test_data)$.pred,
      model = m_name, feature_set = fset_name)

    cat("完成", fset_name, m_name, "\n")
  }
}

cv_results  <- bind_rows(cv_all)
cv_best     <- bind_rows(best_all)
predictions <- bind_rows(pred_all)

write_csv(cv_results,  file.path(OUT, "cv_results_full.csv"))
write_csv(cv_best,     file.path(OUT, "cv_best_per_model.csv"))
write_csv(predictions, file.path(OUT, "predictions_all.csv"))

write_csv(bind_rows(
  tibble(listing_id = train_data$id, property_key = train_data$property_key,
         split = "train", y_true = train_data$price),
  tibble(listing_id = test_data$id,  property_key = test_data$property_key,
         split = "test",  y_true = test_data$price)) %>%
  arrange(listing_id),
  file.path(OUT, "split_record.csv"))

write_csv(bind_rows(
  tibble(listing_id = train_data$id, property_key = train_data$property_key, split = "train"),
  tibble(listing_id = test_data$id,  property_key = test_data$property_key,  split = "test")),
  file.path(OUT, "split_ids.csv"))

pred_met <- predictions %>% group_by(feature_set, model) %>%
  summarise(test_rmse = rmse_vec(y_true, y_pred), test_mae = mae_vec(y_true, y_pred),
            test_rsq = rsq_vec(y_true, y_pred), n = n(), .groups = "drop")

summary_tbl <- cv_best %>%
  select(feature_set, model, penalty, neighbors, cv_rmse, cv_mae, cv_rsq) %>%
  left_join(pred_met, by = c("feature_set","model")) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  arrange(cv_rmse)
write_csv(summary_tbl, file.path(OUT, "model_summary.csv"))

write_csv(tibble(step = c("raw_file", "member1_flags_removed", "modelled"),
                 n    = c(n0, n0 - nrow(dat), nrow(dat)),
                 note = c("apartment_rent_preprocessed.csv",
                          "is_outlier_price / is_outlier_square_feet / is_geo_valid / is_business_error / is_dup_listing",
                          paste0("train ", nrow(train_data), " + test ", nrow(test_data)))),
          file.path(OUT, "sample_flow.csv"))
sink(file.path(OUT, "run_info.txt"))
cat("RUN_LABEL   :", SAMPLE_MODE, "\n")
cat("Run time    :", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Input file  : data/apartment_rent_preprocessed.csv\n")
cat("Input md5   :", unname(tools::md5sum("data/apartment_rent_preprocessed.csv")), "\n")
cat("Rows        : input", n0, "-> modelled", nrow(dat), "\n")
cat("Train / Test:", nrow(train_data), "/", nrow(test_data), "\n")
cat("Seed        :", SEED, "\n")
cat("CV          : 5-fold, grouped by property_key\n")
cat("Grid levels :", GRID_LEVELS, "\n")
cat("Amenities   :", length(amenity_cols), "columns\n\n")
print(sessionInfo())
sink()

cat("\n===== 全部跑完，用时",
    round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1), "min =====\n")
print(as.data.frame(summary_tbl[, c("feature_set","model","penalty","neighbors",
                                    "cv_rmse","cv_mae","cv_rsq","test_rmse")]))
