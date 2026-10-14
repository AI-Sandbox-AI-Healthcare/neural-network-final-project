-- Final project explorer usage report -- run in BigQuery (console.cloud.google.com -> BigQuery).
--
-- 1. The data source is `nn-ui-analytics.analytics_557882182` (in CREATE TEMP TABLE ev,
--    the only place it appears). It is the same GA4 export as the assignment explorers;
--    change it only if the export moves to another project or dataset.
-- 2. Set the date range just below.
-- 3. Paste the whole file into a new BigQuery query tab and Run. Each numbered SELECT
--    returns one table: in the Results panel, click "View results" next to each
--    statement. Save any of them with "Save results".
--
-- "User" = one browser (GA's user_pseudo_id), not a named student: the final project
-- explorer has no student ID, and file names are never sent. Grading stays in D2L.
-- Only final project events are included (assignment = 'final'); test visits
-- (?ga_debug=1 or localhost) are left out.
-- Event and parameter definitions: ANALYTICS.md.

DECLARE start_date STRING DEFAULT '20261001';   -- YYYYMMDD
DECLARE end_date   STRING DEFAULT '20261231';

CREATE TEMP FUNCTION p(params ANY TYPE, k STRING) AS ((
  SELECT COALESCE(value.string_value, CAST(value.int_value AS STRING),
                  CAST(value.double_value AS STRING), CAST(value.float_value AS STRING))
  FROM UNNEST(params) WHERE key = k LIMIT 1));

-- One row per event sent by docs/analytics.js, parameters as columns.
-- analytics.js sends yes/no flags as the strings 'true' / 'false'.
CREATE TEMP TABLE ev AS
SELECT
  user_pseudo_id,
  p(event_params, 'page_load_id')                        AS visit,
  SAFE_CAST(p(event_params, 'seq') AS INT64)             AS seq,
  SAFE_CAST(p(event_params, 't_ms') AS INT64)            AS t_ms,
  TIMESTAMP_MICROS(event_timestamp)                      AS ts,
  event_name,
  p(event_params, 'tab_name')                            AS tab_name,
  p(event_params, 'from_tab')                            AS from_tab,
  p(event_params, 'first_time') = 'true'                 AS first_time,
  -- generic controls
  p(event_params, 'el_id')                               AS el_id,
  p(event_params, 'el_text')                             AS el_text,
  p(event_params, 'from_value')                          AS from_value,
  p(event_params, 'to_value')                            AS to_value,
  -- Dataset tab
  p(event_params, 'file')                                AS file,
  -- Models -> Compare
  p(event_params, 'model_id')                            AS model_id,
  p(event_params, 'selected') = 'true'                   AS selected,
  p(event_params, 'models')                              AS models,
  SAFE_CAST(p(event_params, 'model_count') AS INT64)     AS model_count,
  p(event_params, 'includes_my_model') = 'true'          AS includes_my_model,
  p(event_params, 'metric')                              AS metric,
  -- uploaded results (my_model.csv) and milestones
  SAFE_CAST(p(event_params, 'f1') AS FLOAT64)            AS f1,
  SAFE_CAST(p(event_params, 'auc') AS FLOAT64)           AS auc,
  SAFE_CAST(p(event_params, 'accuracy') AS FLOAT64)      AS accuracy,
  SAFE_CAST(p(event_params, 'precision') AS FLOAT64)     AS precision,
  SAFE_CAST(p(event_params, 'recall') AS FLOAT64)        AS recall,
  SAFE_CAST(p(event_params, 'runs') AS INT64)            AS runs,
  p(event_params, 'f1_band')                             AS f1_band,
  SAFE_CAST(p(event_params, 'elapsed_hr') AS FLOAT64)    AS elapsed_hr,
  SAFE_CAST(p(event_params, 'gpu_hrs') AS FLOAT64)       AS gpu_hrs,
  p(event_params, 'gpu_tier_unlocked') = 'true'          AS gpu_tier_unlocked,
  p(event_params, 'task_complete') = 'true'              AS task_complete,
  p(event_params, 'above_goal') = 'true'                 AS above_goal,
  SAFE_CAST(p(event_params, 'threshold') AS FLOAT64)     AS threshold,
  SAFE_CAST(p(event_params, 'problem_count') AS INT64)   AS problem_count,
  p(event_params, 'first_problem')                       AS first_problem,
  -- running totals (page_hidden)
  SAFE_CAST(p(event_params, 'active_ms') AS INT64)       AS active_ms
FROM `nn-ui-analytics.analytics_557882182.events_*`
WHERE _TABLE_SUFFIX BETWEEN start_date AND end_date
  AND p(event_params, 'page_load_id') IS NOT NULL        -- our events, not GA's automatic ones
  AND p(event_params, 'assignment') = 'final'            -- final project only (the property is shared)
  AND p(event_params, 'debug_mode') IS NULL
  AND NOT REGEXP_CONTAINS(IFNULL(p(event_params, 'page_location'), ''),
                          r'ga_debug|//localhost|//127\.0\.0\.1');

-- Time from each event to the next one on the same page = time spent on that event's
-- tab. A gap that starts at page_hidden is time away (not counted); other gaps are
-- capped at 30 min so a tab left open overnight doesn't count.
CREATE TEMP TABLE gaps AS
SELECT *,
  IF(event_name = 'page_hidden', 0,
     LEAST(LEAD(t_ms) OVER (PARTITION BY visit ORDER BY seq) - t_ms, 30 * 60 * 1000)) AS gap_ms
FROM ev;

-- One row per user: activity totals and milestones, reused by sections 1, 7 and 8.
CREATE TEMP TABLE per_user AS
SELECT
  user_pseudo_id,
  MIN(ts)                                                         AS first_seen,
  COUNT(DISTINCT visit)                                           AS visits,
  COUNT(DISTINCT IF(event_name = 'tab_view', tab_name, NULL))     AS tabs_visited,
  LOGICAL_OR(tab_name = 'Deliverables')                           AS viewed_deliverables,
  COUNTIF(event_name = 'dataset_file_view')                       AS dataset_files_opened,
  COUNTIF(event_name = 'compare_view')                            AS comparisons,
  COUNTIF(event_name = 'template_download')                       AS template_downloads,
  COUNTIF(event_name = 'my_model_upload_failed')                  AS failed_uploads,
  COUNTIF(event_name = 'my_model_uploaded')                       AS accepted_uploads,
  MAX(IF(event_name = 'my_model_uploaded', f1, NULL))             AS best_f1,
  ARRAY_AGG(IF(event_name = 'my_model_uploaded', f1, NULL) IGNORE NULLS
            ORDER BY ts DESC LIMIT 1)[SAFE_OFFSET(0)]             AS latest_f1,
  LOGICAL_OR(event_name = 'milestone_gpu_tier_unlocked')          AS gpu_tier_unlocked,
  LOGICAL_OR(event_name = 'milestone_task_complete')              AS task_complete,
  LOGICAL_OR(event_name = 'milestone_good_performance')           AS above_goal,
  MIN(IF(event_name = 'my_model_uploaded', ts, NULL))             AS first_upload_at,
  MIN(IF(event_name = 'milestone_gpu_tier_unlocked', ts, NULL))   AS gpu_tier_at,
  MIN(IF(event_name = 'milestone_task_complete', ts, NULL))       AS task_complete_at,
  MIN(IF(event_name = 'milestone_good_performance', ts, NULL))    AS above_goal_at
FROM ev
GROUP BY user_pseudo_id;

-- 1. Time per user.
--    visible_minutes: page open and on screen (reading counts, idle >30 min doesn't).
--    active_minutes:  moving / clicking / typing / scrolling within the last 30 s.
WITH per_visit AS (
  SELECT user_pseudo_id, visit,
    SUM(gap_ms) AS visible_ms,
    MAX(IF(event_name = 'page_hidden', active_ms, NULL)) AS active_ms
  FROM gaps
  GROUP BY user_pseudo_id, visit
),
time_per_user AS (
  SELECT user_pseudo_id,
    ROUND(SUM(visible_ms) / 60000, 1) AS visible_minutes,
    ROUND(SUM(active_ms) / 60000, 1) AS active_minutes
  FROM per_visit
  GROUP BY user_pseudo_id
)
SELECT
  user_pseudo_id,
  u.first_seen,
  u.visits,
  t.visible_minutes,
  t.active_minutes,
  u.tabs_visited,
  u.best_f1
FROM per_user u
JOIN time_per_user t USING (user_pseudo_id)
ORDER BY visible_minutes DESC;

-- 2. Time per tab, per user.
SELECT
  user_pseudo_id,
  tab_name,
  ROUND(SUM(gap_ms) / 60000, 1) AS visible_minutes,
  COUNTIF(event_name = 'tab_view') AS times_switched_to
FROM gaps
GROUP BY user_pseudo_id, tab_name
ORDER BY user_pseudo_id, visible_minutes DESC;

-- 3. Tabs, class level: how many users opened each tab and how long they stayed.
--    The first tab of a visit (usually Problem) counts as opened.
WITH per_user_tab AS (
  SELECT user_pseudo_id, tab_name, SUM(gap_ms) / 60000 AS minutes
  FROM gaps
  GROUP BY user_pseudo_id, tab_name
)
SELECT
  tab_name,
  COUNT(*) AS users,
  ROUND(APPROX_QUANTILES(minutes, 2)[OFFSET(1)], 1) AS median_minutes_per_user,
  ROUND(SUM(minutes), 1) AS total_minutes
FROM per_user_tab
GROUP BY tab_name
ORDER BY users DESC;

-- 4. Dataset tab: which CSV files users opened in the Raw Data Explorer.
SELECT
  file,
  COUNT(DISTINCT user_pseudo_id) AS users,
  COUNT(*) AS times_opened
FROM ev
WHERE event_name = 'dataset_file_view'
GROUP BY file
ORDER BY users DESC;

-- 5a. Compare tab: every comparison, in order.
SELECT
  user_pseudo_id,
  ts,
  models,
  model_count,
  includes_my_model
FROM ev
WHERE event_name = 'compare_view'
ORDER BY user_pseudo_id, ts, seq;

-- 5b. Compare tab, class level: how often each model was part of a comparison, and
--     which metric tabs were used.
SELECT 'model' AS kind, model AS name,
  COUNT(DISTINCT user_pseudo_id) AS users, COUNT(*) AS times
FROM ev, UNNEST(SPLIT(models, ',')) AS model
WHERE event_name = 'compare_view'
GROUP BY model
UNION ALL
SELECT 'metric', metric, COUNT(DISTINCT user_pseudo_id), COUNT(*)
FROM ev
WHERE event_name = 'compare_metric'
GROUP BY metric
ORDER BY kind, users DESC;

-- 6a. Every accepted my_model.csv upload (one row per upload), with its scores and the
--     milestones it reached.
SELECT
  user_pseudo_id,
  ts,
  f1, auc, accuracy, precision, recall,
  runs,
  f1_band,
  elapsed_hr, gpu_hrs,
  gpu_tier_unlocked, task_complete, above_goal
FROM ev
WHERE event_name = 'my_model_uploaded'
ORDER BY user_pseudo_id, ts, seq;

-- 6b. Rejected uploads: how often each problem was the first one listed. "Row N:" is
--     removed so the same problem on different rows is counted together.
SELECT
  REGEXP_REPLACE(first_problem, r'^Row \d+: ', '') AS problem,
  COUNT(DISTINCT user_pseudo_id) AS users,
  COUNT(*) AS rejected_uploads,
  ROUND(AVG(problem_count), 1) AS avg_problems_listed
FROM ev
WHERE event_name = 'my_model_upload_failed'
GROUP BY problem
ORDER BY rejected_uploads DESC;

-- 7. Progress and milestones, per user.
--    gpu_tier_unlocked: an uploaded F1 above the best CPU model.
--    task_complete:     an uploaded F1 above the best GPU model.
--    above_goal:        an uploaded F1 above the 0.75 improvement goal.
--    hours_to_*:        from the user's first visit (students may work across days).
SELECT
  user_pseudo_id,
  first_seen,
  visits,
  template_downloads,
  failed_uploads,
  accepted_uploads,
  best_f1,
  latest_f1,
  gpu_tier_unlocked,
  task_complete,
  above_goal,
  ROUND(TIMESTAMP_DIFF(first_upload_at, first_seen, SECOND) / 3600, 1) AS hours_to_first_upload,
  ROUND(TIMESTAMP_DIFF(task_complete_at, first_seen, SECOND) / 3600, 1) AS hours_to_task_complete,
  ROUND(TIMESTAMP_DIFF(above_goal_at, first_seen, SECOND) / 3600, 1) AS hours_to_above_goal
FROM per_user
ORDER BY best_f1 DESC NULLS LAST, user_pseudo_id;

-- 8. Class summary: a funnel from first visit to the 0.75 goal. Milestone percentages
--    are of users with at least one accepted upload.
SELECT
  COUNT(*) AS users,
  COUNTIF(viewed_deliverables) AS viewed_deliverables,
  COUNTIF(template_downloads > 0) AS downloaded_template,
  COUNTIF(failed_uploads > 0) AS had_rejected_upload,
  COUNTIF(accepted_uploads > 0) AS uploaded_results,
  COUNTIF(gpu_tier_unlocked) AS gpu_tier_unlocked,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(gpu_tier_unlocked), COUNTIF(accepted_uploads > 0)), 1) AS pct_gpu_tier_unlocked,
  COUNTIF(task_complete) AS task_complete,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(task_complete), COUNTIF(accepted_uploads > 0)), 1) AS pct_task_complete,
  COUNTIF(above_goal) AS above_goal,
  ROUND(100 * SAFE_DIVIDE(COUNTIF(above_goal), COUNTIF(accepted_uploads > 0)), 1) AS pct_above_goal,
  ROUND(APPROX_QUANTILES(best_f1, 2 IGNORE NULLS)[OFFSET(1)], 3) AS median_best_f1,
  ROUND(MAX(best_f1), 3) AS top_best_f1
FROM per_user;

-- 9. Score distribution: users by the 0.05-wide band of their best uploaded F1.
SELECT
  CONCAT(FORMAT('%.2f', FLOOR(best_f1 * 20) / 20), '-',
         FORMAT('%.2f', FLOOR(best_f1 * 20) / 20 + 0.05)) AS best_f1_band,
  COUNT(*) AS users,
  COUNTIF(task_complete) AS of_which_task_complete
FROM per_user
WHERE best_f1 IS NOT NULL
GROUP BY best_f1_band
ORDER BY best_f1_band;
