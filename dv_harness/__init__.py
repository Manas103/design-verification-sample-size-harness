"""Reliability and design-verification reporting harness.

Five independent modules, each usable on its own:

attribute_sampling   -- binomial (pass/fail) reliability demonstration sample size
variables_sampling   -- normal-distribution (variables) tolerance-interval sample size
gage_rr              -- simulated crossed gage R&R study, ANOVA method, pass/fail rule
reliability          -- right-censored Weibull life-data fitting, B10 life with confidence bounds, failure-mode Pareto
requirements_report  -- per-requirement VERIFIED / NOT VERIFIED report with refusal on missing evidence
"""
