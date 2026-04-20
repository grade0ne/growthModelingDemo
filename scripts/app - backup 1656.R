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

fit_with_trace <- function(times, y_obs, y0, start_r, start_alpha, maxit = 100) {
  path_env <- new.env(parent = emptyenv())
  path_env$path <- tibble(step = integer(), r = numeric(), alpha = numeric(), sse = numeric(), valid = logical())
  
  resid_fn <- residual_objective_factory(times, y_obs, y0, path_env)
  
  fit <- minpack.lm::nls.lm(
    par = c(start_r, start_alpha),
    fn = resid_fn,
    lower = c(1e-6, 1e-6),
    upper = c(10, 10),
    control = minpack.lm::nls.lm.control(maxiter = maxit)
  )
  
  best_resid <- resid_fn(fit$par)
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
  
  path <- path_all %>%
    mutate(frame = row_number())
  
  list(
    fit = fit,
    path = path,
    best_r = fit$par[1],
    best_alpha = fit$par[2],
    best_sse = best_sse
  )
}

ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$style(HTML("\n      .anim-controls .btn {\n        font-size: 18px;\n        padding: 10px 18px;\n        font-weight: 600;\n        display: inline-block;\n        margin-bottom: 10px;\n      }\n      .speed-wrap {\n        width: 100%;\n      }\n      .speed-wrap .irs {\n        width: 100%;\n      }\n      .speed-labels {\n        width: 100%;\n        display: flex;\n        justify-content: space-between;\n        box-sizing: border-box;\n        padding: 0 16px;\n        font-size: 12px;\n        color: #666;\n        margin-top: -8px;\n        margin-bottom: 10px;\n      }\n      details.advanced-panel {\n        margin-top: 14px;\n        padding: 8px 10px;\n        border: 1px solid #ddd;\n        border-radius: 6px;\n        background: #fafafa;\n      }\n      details.advanced-panel summary {\n        cursor: pointer;\n        font-weight: 600;\n      }\n      .param-grid {\n        display: grid;\n        grid-template-columns: auto auto;\n        gap: 6px 18px;\n        width: fit-content;\n        margin-top: 8px;\n      }\n      .equation-wrap {
        font-size: 18px;
        line-height: 1.6;
      }
      #refit.btn-warning:not([disabled]) {
        background-color: #d4a017;
        border-color: #b38712;
        color: #fff;
      }
      #refit[disabled],
      #refit.btn.disabled {
        background-color: #bdbdbd;
        border-color: #a6a6a6;
        color: #fff;
        opacity: 1;
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
        actionButton("refit", "Run fit", class = "btn-warning"),
        actionButton("regen", "Randomize points")
      ),
      h4("Generate data"),
      sliderInput("r_true", "True r", min = 0.05, max = 2.5, value = 0.9, step = 0.01),
      sliderInput("alpha_true", "True alpha", min = 0.001, max = 0.25, value = 0.03, step = 0.001),
      sliderInput("noise_sd", "Variance", min = 0, max = 15, value = 4, step = 0.5),
      sliderInput("sample_size", "Sample size", min = 5, max = 100, value = 25, step = 1),
      tags$hr(),
      
      h4("Starting guess for fitting"),
      sliderInput("r_start", "Initial guess: r", min = 0.05, max = 2.5, value = 0.3, step = 0.01),
      sliderInput("alpha_start", "Initial guess: alpha", min = 0.001, max = 0.25, value = 0.08, step = 0.001)
    ),
    
    column(
      width = 6,
      div(style = "text-align:center; margin-bottom:12px;", uiOutput("true_equation")),
      plotOutput("curve_plot", height = "460px"),
      br(),
      fluidRow(
        column(6, plotOutput("sse_trace_plot", height = "220px")),
        column(6, plotOutput("param_bar_plot", height = "220px"))
      ),
      tags$details(
        open = NULL,
        class = "advanced-panel",
        tags$summary("Read me"),
        tags$p("This demo shows how a logistic growth model is fit to noisy data."),
        tags$p("Use the sliders on the left to set the true data-generating parameters and the starting guess for the optimizer."),
        tags$p("Click Run fit to watch the optimizer move through trial parameter values. The main plot shows the data and model curves, while the lower plots show error reduction and parameter movement across the fitting process."),
        tags$p("If you change the data or fitting inputs after a run, the previous final fit is treated as stale until you run the fit again.")
      )
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
        class = "advanced-panel",
        tags$summary("Advanced animation settings"),
        sliderInput("curve_points", "Points per curve", min = 10, max = 300, value = 40, step = 5),
        tags$p(
          style = "font-size: 0.9em; color: #555; margin-bottom: 0;",
          "Fewer points per curve make the animation much smoother."
        )
      )
    )
  )
)

server <- function(input, output, session) {
  y0_fixed <- 5
  times <- reactive(seq(0, 12, length.out = input$sample_size))
  dataset_seed <- reactiveVal(sample.int(1e8, 1))
  fit_is_current <- reactiveVal(FALSE)
  
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
      fit_is_current(FALSE)
      shinyjs::enable("refit")
    },
    ignoreInit = TRUE
  )
  
  fit_results <- eventReactive(input$refit, {
    dat <- dataset()
    fit_with_trace(
      times = dat$time,
      y_obs = dat$observed,
      y0 = y0_fixed,
      start_r = input$r_start,
      start_alpha = input$alpha_start,
      maxit = 100
    )
  }, ignoreInit = TRUE)
  
  observeEvent(input$refit, {
    fit_is_current(TRUE)
    shinyjs::disable("refit")
  }, ignoreInit = TRUE)
  
  session$onFlushed(function() {
    shinyjs::click("refit")
  }, once = TRUE)
  
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
    plot_times <- seq(min(dat$time), max(dat$time), length.out = input$curve_points)
    
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
    
    if ("final" %in% input$line_toggle && final_frame && isTRUE(fit_is_current())) {
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
    
    if (isTRUE(fit_is_current())) {
      final_rmse <- sqrt(fit$best_sse / n)
      abline(h = final_rmse, col = "goldenrod2", lwd = 2, lty = 2)
    }
  })
  
  output$param_bar_plot <- renderPlot({
    fit <- fit_results()
    path <- fit$path
    state <- current_state()
    
    current_vals <- c(r = state$r, alpha = state$alpha)
    true_vals <- c(r = input$r_true, alpha = input$alpha_true)
    init_vals <- c(r = input$r_start, alpha = input$alpha_start)
    fit_vals <- c(r = fit$best_r, alpha = fit$best_alpha)
    
    calc_ylim <- function(true_val, init_val, current_val, fit_val, show_fit) {
      ref_vals <- c(true_val, init_val, current_val)
      if (show_fit) ref_vals <- c(ref_vals, fit_val)
      max_dev <- max(abs(ref_vals - true_val), na.rm = TRUE)
      if (!is.finite(max_dev) || max_dev == 0) max_dev <- max(abs(true_val) * 0.05, 0.01)
      buffer <- max(0.08 * max_dev, 0.01)
      c(true_val - max_dev - buffer, true_val + max_dev + buffer)
    }
    
    ylim_r <- calc_ylim(true_vals["r"], init_vals["r"], current_vals["r"], fit_vals["r"], isTRUE(fit_is_current()))
    ylim_a <- calc_ylim(true_vals["alpha"], init_vals["alpha"], current_vals["alpha"], fit_vals["alpha"], isTRUE(fit_is_current()))
    
    dev_r_scale <- max(abs(c(init_vals["r"], current_vals["r"], if (isTRUE(fit_is_current())) fit_vals["r"] else NA) - true_vals["r"]), na.rm = TRUE)
    dev_a_scale <- max(abs(c(init_vals["alpha"], current_vals["alpha"], if (isTRUE(fit_is_current())) fit_vals["alpha"] else NA) - true_vals["alpha"]), na.rm = TRUE)
    if (!is.finite(dev_r_scale) || dev_r_scale == 0) dev_r_scale <- 1
    if (!is.finite(dev_a_scale) || dev_a_scale == 0) dev_a_scale <- 1
    
    alpha_r <- max(0.02, min(1, abs(current_vals["r"] - true_vals["r"]) / dev_r_scale))
    alpha_a <- max(0.02, min(1, abs(current_vals["alpha"] - true_vals["alpha"]) / dev_a_scale))
    col_r <- adjustcolor("red", alpha.f = alpha_r)
    col_a <- adjustcolor("red", alpha.f = alpha_a)
    
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par))
    par(mfrow = c(1, 2), mar = c(4.2, 4.8, 4.2, 1), mgp = c(2.4, 0.8, 0))
    
    plot(
      0, 0,
      type = "n",
      xlim = c(0.5, 1.5), ylim = ylim_r,
      xaxt = "n", xlab = "",
      ylab = "r",
      main = sprintf("r = %.4f", current_vals["r"]),
      adj = 0
    )
    abline(h = true_vals["r"], col = "blue", lwd = 2)
    abline(h = init_vals["r"], col = "purple", lwd = 2)
    rect(0.75, true_vals["r"], 1.25, current_vals["r"], border = "black", lwd = 2, col = col_r)
    if (isTRUE(fit_is_current())) {
      segments(0.75, fit_vals["r"], 1.25, fit_vals["r"], col = "goldenrod2", lwd = 2)
    }
    
    plot(
      0, 0,
      type = "n",
      xlim = c(0.5, 1.5), ylim = ylim_a,
      xaxt = "n", xlab = "",
      ylab = expression(alpha),
      main = sprintf("alpha = %.4f", current_vals["alpha"]),
      adj = 0
    )
    abline(h = true_vals["alpha"], col = "blue", lwd = 2)
    abline(h = init_vals["alpha"], col = "purple", lwd = 2)
    rect(0.75, true_vals["alpha"], 1.25, current_vals["alpha"], border = "black", lwd = 2, col = col_a)
    if (isTRUE(fit_is_current())) {
      segments(0.75, fit_vals["alpha"], 1.25, fit_vals["alpha"], col = "goldenrod2", lwd = 2)
    }
  })
}

shinyApp(ui, server)
