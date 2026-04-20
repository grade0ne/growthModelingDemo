library(shiny)
library(shinyjs)
library(dplyr)
library(tidyr)
library(purrr)
library(scales)
library(withr)
library(minpack.lm)

# Logistic model in the same closed-form used by the custom model:
# y(t) = (r * y0) / (alpha * y0 + (r - alpha * y0) * exp(-r * t))
logistic_r_alpha <- function(t, y0, r, alpha) {
  (r * y0) / (alpha * y0 + (r - alpha * y0) * exp(-r * t))
}

simulate_data <- function(times, y0, r, alpha, noise_sd) {
  truth <- logistic_r_alpha(times, y0, r, alpha)
  observed <- pmax(0, truth + rnorm(length(times), mean = 0, sd = noise_sd))
  tibble(time = times, truth = truth, observed = observed)
}

residual_objective_factory <- function(times, y_obs, y0, path_env) {
  force(times); force(y_obs); force(y0); force(path_env)
  
  function(par) {
    r <- par[1]
    alpha <- par[2]
    
    if (!is.finite(r) || !is.finite(alpha) || r <= 0 || alpha <= 0) {
      resid <- rep(1e6, length(y_obs))
      sse <- sum(resid^2)
      path_env$path <- bind_rows(
        path_env$path,
        tibble(step = nrow(path_env$path) + 1, r = r, alpha = alpha, sse = sse, valid = FALSE)
      )
      return(resid)
    }
    
    pred <- logistic_r_alpha(times, y0, r, alpha)
    resid <- y_obs - pred
    sse <- sum(resid^2)
    
    path_env$path <- bind_rows(
      path_env$path,
      tibble(step = nrow(path_env$path) + 1, r = r, alpha = alpha, sse = sse, valid = TRUE)
    )
    resid
  }
}

fit_with_trace <- function(times, y_obs, y0, start_r, start_alpha, maxit = 100, method = "LM") {
  path_env <- new.env(parent = emptyenv())
  path_env$path <- tibble(step = integer(), r = numeric(), alpha = numeric(), sse = numeric(), valid = logical())
  
  resid_fn <- residual_objective_factory(times, y_obs, y0, path_env)
  
  if (method == "LM") {
    fit <- minpack.lm::nls.lm(
      par = c(start_r, start_alpha),
      fn = resid_fn,
      lower = c(1e-6, 1e-6),
      upper = c(10, 10),
      control = minpack.lm::nls.lm.control(maxiter = maxit)
    )
    best_par <- fit$par
  } else {
    obj_fn <- function(par) {
      resid <- resid_fn(par)
      sum(resid^2)
    }
    fit <- optim(
      par = c(start_r, start_alpha),
      fn = obj_fn,
      method = "L-BFGS-B",
      lower = c(1e-6, 1e-6),
      upper = c(10, 10),
      control = list(maxit = maxit)
    )
    best_par <- fit$par
  }
  
  best_resid <- resid_fn(best_par)
  best_sse <- sum(best_resid^2)
  
  path_all <- path_env$path %>%
    distinct(r, alpha, .keep_all = TRUE)
  
  if (nrow(path_all) == 0 ||
      isTRUE(abs(path_all$r[1] - start_r) > .Machine$double.eps^0.5) ||
      isTRUE(abs(path_all$alpha[1] - start_alpha) > .Machine$double.eps^0.5)) {
    start_pred <- logistic_r_alpha(times, y0, start_r, start_alpha)
    start_sse <- sum((y_obs - start_pred)^2)
    start_row <- tibble(step = 0, r = start_r, alpha = start_alpha, sse = start_sse, valid = TRUE)
    path_all <- bind_rows(start_row, path_all)
  }
  
  path <- path_all %>% mutate(frame = row_number())
  
  list(
    fit = fit,
    path = path,
    best_r = best_par[1],
    best_alpha = best_par[2],
    best_sse = best_sse
  )
}

ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$style(HTML("\n      .anim-controls .btn {\n        font-size: 18px;\n        padding: 10px 18px;\n        font-weight: 600;\n        display: inline-block;\n        margin-bottom: 10px;\n      }\n      .speed-wrap {\n        width: 100%;\n      }\n      .speed-wrap .irs {\n        width: 100%;\n      }\n      .speed-labels {\n        width: 100%;\n        display: flex;\n        justify-content: space-between;\n        box-sizing: border-box;\n        padding: 0 16px;\n        font-size: 12px;\n        color: #666;\n        margin-top: -8px;\n        margin-bottom: 10px;\n      }\n      details.advanced-panel {\n        margin-top: 14px;\n        padding: 8px 10px;\n        border: 1px solid #ddd;\n        border-radius: 6px;\n        background: #fafafa;\n      }\n      details.advanced-panel summary {\n        cursor: pointer;\n        font-weight: 600;\n      }\n      .param-grid {\n        display: grid;\n        grid-template-columns: auto auto;\n        gap: 6px 18px;\n        width: fit-content;\n        margin-top: 8px;\n      }\n      .equation-wrap {
        font-size: 18px;
        line-height: 1.6;
      }\n    ")),
    tags$script(HTML("\n      Shiny.addCustomMessageHandler('playAnimation', function(msg) {\n        var sliderEl = $('#' + msg.id);\n        var slider = sliderEl.data('ionRangeSlider');\n        if (!slider) return;\n\n        if (window.shinyPlayTimer) {\n          clearInterval(window.shinyPlayTimer);\n        }\n\n        var min = Number(slider.result.min);\n        var max = Number(slider.result.max);\n        var step = Number(slider.result.step) || 1;\n        var current = msg.restart ? min : Number(slider.result.from);\n\n        function pushValue(val) {\n          slider.update({ from: val });\n          Shiny.setInputValue(msg.id, val, { priority: 'event' });\n          sliderEl.trigger('change');\n          sliderEl.trigger('change.irs');\n        }\n\n        pushValue(current);\n\n        window.shinyPlayTimer = setInterval(function() {\n          current = current + step;\n          if (current >= max) {\n            pushValue(max);\n            clearInterval(window.shinyPlayTimer);\n            window.shinyPlayTimer = null;\n          } else {\n            pushValue(current);\n          }\n        }, msg.delay);\n      });\n\n      Shiny.addCustomMessageHandler('pauseAnimation', function(msg) {\n        if (window.shinyPlayTimer) {\n          clearInterval(window.shinyPlayTimer);\n          window.shinyPlayTimer = null;\n        }\n      });\n    ")),
    tags$script(src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js")
  ),
  titlePanel("How model fitting works: an animated r-alpha logistic demo"),
  
  fluidRow(
    column(
      width = 3,
      tags$hr(),
      
      div(
        style = "display:flex; gap:8px; align-items:center; margin-bottom:12px;",
        actionButton("regen", "Randomize points")
      ),
      h4("Generate data"),
      sliderInput("r_true", "True r", min = 0.05, max = 2.5, value = 0.9, step = 0.01),
      sliderInput("alpha_true", "True alpha", min = 0.001, max = 0.25, value = 0.03, step = 0.001),
      sliderInput("noise_sd", "Variance", min = 0, max = 15, value = 4, step = 0.5),
      sliderInput("sample_size", "Sample frequency", min = 5, max = 100, value = 25, step = 1),
      tags$hr(),
      
      h4("Starting guess for fitting"),
      selectInput("optimizer", "Optimizer", choices = c("Levenberg-Marquardt" = "LM", "L-BFGS-B" = "LBFGSB"), selected = "LM"),
      sliderInput("r_start", "Initial guess: r", min = 0.05, max = 2.5, value = 0.3, step = 0.01),
      sliderInput("alpha_start", "Initial guess: alpha", min = 0.001, max = 0.25, value = 0.08, step = 0.001)
    ),
    
    column(
      width = 6,
      div(style = "text-align:center; margin-bottom:12px;", uiOutput("true_equation")),
      plotOutput("curve_plot", height = "460px"),
      br(),
      fluidRow(
        column(8, plotOutput("sse_trace_plot", height = "260px")),
        column(4, plotOutput("rmse_surface_plot", height = "260px"))
      ),
      tags$hr(),
      h4("Log"),
      actionButton("log_model", "Log model"),
      br(),
      br(),
      div(class = "log-table", tableOutput("model_log_table"))
    ),
    
    column(
      width = 3,
      h4("Animation controls"),
      div(class = "anim-controls", uiOutput("frame_ui")),
      div(
        class = "speed-wrap",
        sliderInput("anim_speed_scale", "Animation speed", min = 1, max = 5, value = 3, step = 1, ticks = FALSE)
      ),
      div(class = "speed-labels", span("Slow"), span("Fast")),
      checkboxGroupInput(
        "line_toggle",
        "Show lines",
        choices = c(
          "True curve" = "true",
          "Current guess" = "guess",
          "Final fit" = "final",
          "Residual segments" = "residuals"
        ),
        selected = c("true", "guess", "final")
      ),
      tags$details(
        open = NULL,
        class = "advanced-panel",
        tags$summary("Read me"),
        tags$p("This demo shows how a logistic growth model is fit to noisy data."),
        tags$p("Use the sliders on the left to set the true data-generating parameters and the starting guess for the optimizer."),
        tags$p("The main plot shows the data and model curves, while the lower plots show error reduction and parameter movement across the fitting process."),
        tags$p("Use Log model to save the current settings and fitted values to the table in the center panel.")
      )
    )
  )
)

server <- function(input, output, session) {
  y0_fixed <- 5
  times <- reactive(seq(0, 12, length.out = input$sample_size))
  dataset_seed <- reactiveVal(sample.int(1e8, 1))
  model_log <- reactiveVal(
    tibble(
      row_id = integer(),
      true_r = numeric(),
      true_a = numeric(),
      var = numeric(),
      init_r = numeric(),
      init_a = numeric(),
      fitted_r = numeric(),
      fitted_a = numeric()
    )
  )
  
  speed_ms <- reactive({
    unname(as.numeric(c(`1` = 500, `2` = 400, `3` = 300, `4` = 200, `5` = 100)[as.character(input$anim_speed_scale)]))
  })
  
  observeEvent(input$regen, {
    dataset_seed(sample.int(1e8, 1))
  })
  
  dataset <- reactive({
    with_seed(dataset_seed(), {
      simulate_data(
        times = times(),
        y0 = y0_fixed,
        r = input$r_true,
        alpha = input$alpha_true,
        noise_sd = input$noise_sd
      )
    })
  })
  
  observeEvent(
    list(
      input$r_true,
      input$alpha_true,
      input$noise_sd,
      input$sample_size,
      input$r_start,
      input$alpha_start,
      input$regen
    ),
    {
      updateSliderInput(session, "frame", value = 1)
    },
    ignoreInit = TRUE
  )
  
  fit_results <- reactive({
    dat <- dataset()
    fit_with_trace(
      times = dat$time,
      y_obs = dat$observed,
      y0 = y0_fixed,
      start_r = input$r_start,
      start_alpha = input$alpha_start,
      maxit = 100,
      method = input$optimizer
    )
  })
  
  output$frame_ui <- renderUI({
    n <- nrow(fit_results()$path)
    tagList(
      div(
        style = "text-align:left; margin-bottom:10px;",
        actionButton("play_anim", "Play", class = "btn-primary"),
        tags$span(style = "display:inline-block; width:8px;"),
        actionButton("pause_anim", "⏸"),
        tags$span(style = "display:inline-block; width:8px;"),
        actionButton("reset_anim", "↺")
      ),
      sliderInput(
        "frame",
        "Trial position",
        min = 1,
        max = max(1, n),
        value = max(1, n),
        step = 1,
        animate = FALSE
      )
    )
  })
  
  observeEvent(input$play_anim, {
    path <- fit_results()$path
    current_frame <- input$frame %||% nrow(path)
    restart_now <- current_frame >= nrow(path)
    
    session$sendCustomMessage("playAnimation", list(
      id = "frame",
      delay = speed_ms(),
      restart = restart_now
    ))
  })
  
  observeEvent(input$pause_anim, {
    session$sendCustomMessage("pauseAnimation", list(id = "frame"))
  })
  
  observeEvent(input$reset_anim, {
    session$sendCustomMessage("pauseAnimation", list(id = "frame"))
    updateSliderInput(session, "frame", value = 1)
  })
  
  current_state <- reactive({
    path <- fit_results()$path
    req(nrow(path) >= 1)
    frame <- min(input$frame %||% nrow(path), nrow(path))
    path[frame, ]
  })
  
  output$true_equation <- renderUI({
    withMathJax(HTML("$$ \\frac{dN}{dt} = (r - \\alpha N)N $$"))
  })
  
  curve_cache <- reactive({
    dat <- dataset()
    fit <- fit_results()
    plot_times <- seq(min(dat$time), max(dat$time), length.out = 50)
    
    true_curve <- tibble(
      time = plot_times,
      N = logistic_r_alpha(plot_times, y0_fixed, input$r_true, input$alpha_true),
      curve = "true"
    )
    
    guess_curves <- fit$path %>%
      select(frame, r, alpha, sse) %>%
      mutate(curve_data = purrr::map2(r, alpha, ~ tibble(
        time = plot_times,
        N = logistic_r_alpha(plot_times, y0_fixed, .x, .y)
      ))) %>%
      tidyr::unnest(curve_data)
    
    final_curve <- tibble(
      time = plot_times,
      N = logistic_r_alpha(plot_times, y0_fixed, fit$best_r, fit$best_alpha),
      curve = "final"
    )
    
    x_rng <- range(dat$time, na.rm = TRUE)
    y_rng <- range(c(dat$observed, dat$truth), na.rm = TRUE)
    y_pad <- 0.08 * diff(y_rng)
    if (!is.finite(y_pad) || y_pad == 0) y_pad <- 1
    y_limits <- c(max(0, y_rng[1] - y_pad), y_rng[2] + y_pad)
    
    list(
      dat = dat,
      fit = fit,
      true_curve = true_curve,
      guess_curves = guess_curves,
      final_curve = final_curve,
      x_rng = x_rng,
      y_limits = y_limits
    )
  })
  output$curve_plot <- renderPlot({
    cache <- curve_cache()
    dat <- cache$dat
    fit <- cache$fit
    path <- fit$path
    state <- current_state()
    frame_now <- input$frame %||% nrow(path)
    final_frame <- frame_now >= nrow(path)
    
    par(mar = c(4.2, 4.4, 3.8, 0.8))
    
    plot(
      dat$time,
      dat$observed,
      xlim = cache$x_rng,
      ylim = cache$y_limits,
      xlab = "Time",
      ylab = "Population size",
      pch = 16,
      cex = 1,
      xaxs = "i",
      yaxs = "i"
    )
    
    if ("true" %in% input$line_toggle) {
      lines(dat$time, dat$truth, col = "blue", lty = 2, lwd = 1.5)
      lines(cache$true_curve$time, cache$true_curve$N, col = "blue", lwd = 2)
    }
    
    if ("guess" %in% input$line_toggle && !final_frame) {
      guess_frame <- cache$guess_curves[cache$guess_curves$frame == state$frame, ]
      if (nrow(guess_frame) > 0) {
        lines(guess_frame$time, guess_frame$N, col = "red", lwd = 2)
      }
    }
    
    if ("final" %in% input$line_toggle && final_frame) {
      lines(cache$final_curve$time, cache$final_curve$N, col = "goldenrod2", lwd = 2)
    }
    
    if ("residuals" %in% input$line_toggle) {
      if (final_frame) {
        pred_vals <- logistic_r_alpha(dat$time, y0_fixed, fit$best_r, fit$best_alpha)
      } else {
        pred_vals <- logistic_r_alpha(dat$time, y0_fixed, state$r, state$alpha)
      }
      segments(dat$time, dat$observed, dat$time, pred_vals, col = "darkgreen", lwd = 1)
    }
  })
  
  output$sse_trace_plot <- renderPlot({
    fit <- fit_results()
    path <- fit$path
    current_frame <- min(input$frame %||% nrow(path), nrow(path))
    
    n <- length(dataset()$observed)
    rmse_vals <- sqrt(path$sse / n)
    
    finite_vals <- rmse_vals[is.finite(rmse_vals)]
    if (length(finite_vals) == 0) return(invisible(NULL))
    
    q_lo <- as.numeric(stats::quantile(finite_vals, probs = 0.05, na.rm = TRUE, names = FALSE))
    q_hi <- as.numeric(stats::quantile(finite_vals, probs = 0.95, na.rm = TRUE, names = FALSE))
    pad <- 0.05 * (q_hi - q_lo)
    if (!is.finite(pad) || pad <= 0) pad <- 0.1
    y_lim <- c(q_lo - pad, q_hi + pad)
    
    par(mar = c(4.2, 4.8, 3.0, 0.8))
    plot(
      path$frame,
      rmse_vals,
      type = "l",
      lwd = 2,
      xlab = "Evaluation",
      ylab = "RMSE",
      main = "RMSE across evaluations",
      ylim = y_lim
    )
    points(path$frame[current_frame], rmse_vals[current_frame], pch = 16, cex = 1.2, col = "red")
    
    final_rmse <- sqrt(fit$best_sse / n)
    abline(h = final_rmse, col = "goldenrod2", lwd = 2, lty = 2)
  })
  
  output$rmse_surface_plot <- renderPlot({
    dat <- dataset()
    fit <- fit_results()
    path <- fit$path
    state <- current_state()
    current_frame <- min(input$frame %||% nrow(path), nrow(path))
    path_now <- path[seq_len(current_frame), , drop = FALSE]
    
    r_center <- c(input$r_true, input$r_start, fit$best_r, path$r)
    a_center <- c(input$alpha_true, input$alpha_start, fit$best_alpha, path$alpha)
    
    r_range <- range(r_center, na.rm = TRUE)
    a_range <- range(a_center, na.rm = TRUE)
    r_span <- diff(r_range)
    a_span <- diff(a_range)
    if (!is.finite(r_span) || r_span <= 0) r_span <- 0.25
    if (!is.finite(a_span) || a_span <= 0) a_span <- 0.025
    
    r_mid <- mean(r_range)
    a_mid <- mean(a_range)
    span <- max(r_span, a_span)
    pad <- 0.2 * span
    half_span <- 0.5 * span + pad
    
    r_min <- max(0.01, r_mid - half_span)
    r_max <- min(2.5, r_mid + half_span)
    a_min <- max(1e-4, a_mid - half_span)
    a_max <- min(0.25, a_mid + half_span)
    
    if (!is.finite(r_min) || !is.finite(r_max) || r_min >= r_max) {
      r_min <- 0.05
      r_max <- 2.5
    }
    if (!is.finite(a_min) || !is.finite(a_max) || a_min >= a_max) {
      a_min <- 0.001
      a_max <- 0.25
    }
    
    r_grid <- seq(r_min, r_max, length.out = 60)
    a_grid <- seq(a_min, a_max, length.out = 60)
    
    rmse_mat <- outer(r_grid, a_grid, Vectorize(function(r, alpha) {
      pred <- logistic_r_alpha(dat$time, y0_fixed, r, alpha)
      sqrt(mean((dat$observed - pred)^2))
    }))
    
    par(mar = c(4.2, 4.8, 3.0, 1.2), bg = "white", pty = "s")
    
    # no heatmap; just a square parameter-space plot
    plot(NA, NA,
         xlim = c(r_min, r_max),
         ylim = c(a_min, a_max),
         xlab = "r",
         ylab = expression(alpha),
         main = "Parameter space",
         type = "n"
    )
    box()
    
    if (nrow(path_now) >= 2) {
      lines(path_now$r, path_now$alpha, lwd = 2, col = "black")
    }
    points(path_now$r, path_now$alpha, pch = 16, cex = 0.4, col = adjustcolor("black", alpha.f = 0.4))
    
    points(state$r, state$alpha, pch = 16, cex = 1.2, col = "red")
    points(input$r_true, input$alpha_true, pch = 16, cex = 1.3, col = "blue")
    points(input$r_start, input$alpha_start, pch = 16, cex = 1.2, col = "green3")
    points(fit$best_r, fit$best_alpha, pch = 16, cex = 1.3, col = "goldenrod2")
    
    legend(
      "topright",
      legend = c("True", "Start", "Current", "Final"),
      pch = 16,
      col = c("blue", "green3", "red", "goldenrod2"),
      pt.cex = c(1.3, 1.2, 1.2, 1.3),
      bty = "n",
      cex = 0.85
    )
  })
  
  observeEvent(input$log_model, {
    fit <- fit_results()
    old_log <- model_log()
    new_row <- tibble(
      row_id = nrow(old_log) + 1,
      true_r = input$r_true,
      true_a = input$alpha_true,
      var = input$noise_sd,
      init_r = input$r_start,
      init_a = input$alpha_start,
      fitted_r = fit$best_r,
      fitted_a = fit$best_alpha
    )
    model_log(bind_rows(old_log, new_row))
  })
  
  output$model_log_table <- renderTable({
    log_df <- model_log()
    if (nrow(log_df) == 0) return(NULL)
    
    log_df %>%
      transmute(
        ` ` = row_id,
        `true r` = round(true_r, 4),
        `true a` = round(true_a, 4),
        var = round(var, 4),
        `init r` = round(init_r, 4),
        `init a` = round(init_a, 4),
        `fitted r` = round(fitted_r, 4),
        `fitted a` = round(fitted_a, 4)
      )
  }, striped = TRUE, bordered = TRUE, spacing = "s", width = "100%")
}

shinyApp(ui, server)
