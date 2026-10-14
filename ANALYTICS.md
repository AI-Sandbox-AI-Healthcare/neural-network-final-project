# Usage analytics (GitHub Pages explorer)

The published final project explorer sends interaction events to **Google Analytics 4**
through one hand-maintained file, [`docs/analytics.js`](docs/analytics.js). It mirrors
`docs/analytics.js` in **neural-network-assignments-ui-explorer**: same GA4 property
(`G-T41SB4WQ70`), same host check, consent setting, opt-out and generic events. That
means all explorers can be analysed together.

- `build_docs.py` adds `<script src="analytics.js" defer></script>` to `docs/index.html`.
- The local Flask explorer (`run_explorer.py`) never loads it, so nothing is tracked
  when students run it on their own machines.
- The explorer code is not modified. Once the page's own scripts have run,
  `analytics.js` wraps their global functions (`toggleModel`, `goToCompare`,
  `switchRawTab`, ...) and listens at the document level for clicks, changes,
  scrolling, copying, visibility and errors. If a function is renamed in
  `run_explorer.py`, only that one event stops; nothing breaks.
- Every event carries `assignment = final`, so it can be separated from the assignment
  explorers (`a1`, `a2`, `a3`, `landing`) in the same property.

## Setup

The GA4 property and settings are shared with the assignment explorers, so no new
property or data stream is needed. Pages on `ai-sandbox-ai-healthcare.github.io` send
to it; the stream URL is only a label.

1. **Test locally** before pushing: run `python build_docs.py`, then
   `python -m http.server 8000 -d docs`, and open <http://localhost:8000/?ga_debug=1>.
   Every event is logged in the browser console as `[ga] <name> {…}` and shows up
   within seconds under GA4 Admin → DebugView.
2. **Commit and push** `docs/` (both `index.html` and `analytics.js`). GitHub Pages
   redeploys. Check Reports → Realtime after you open the live site.
3. **Opt yourself out.** Open the live page once with `?notrack=1` in every browser and
   device you use. The opt-out is shared with the assignment explorers (same site, same
   storage key); `?notrack=0` undoes it.
4. **Register the new custom definitions** (Admin → Custom definitions). The ones the
   assignment explorers already use (`assignment`, `tab_name`, `from_tab`, `el_id`,
   `el_text`, `el_action`, `scroller`, `f1`, `auc`, `accuracy`, the `*_ms` metrics) are
   reused as they are.
   - Event-scoped dimensions: `model_id`, `models`, `metric`, `file`, `f1_band`,
     `gpu_tier_unlocked`, `task_complete`, `above_goal`.
   - Custom metrics (standard unit): `precision`, `recall`, `threshold`, `runs`,
     `model_count`, `problem_count`.
   - Don't register `first_problem`, `page_load_id`, `t_ms` or `seq`. Their high
     cardinality would only produce "(other)" rows; BigQuery still gets them.

## Events

Every event carries `assignment` (`final`), `tab_name` (the active tab, e.g. `Models`),
`page_load_id`, `t_ms` (ms since page load) and `seq` (order within the page load).

| Event | When | Extra parameters |
|---|---|---|
| `page_view` | page load (automatic) | — |
| `client_info` | page load | `viewport_w`, `viewport_h`, `pixel_ratio`, `touch`, `dark_mode`, `load_ms` |
| `tab_view` | tab switched | `from_tab`, `from_tab_ms`, `from_tab_active_ms`, `first_time` |
| `dataset_file_view` | a CSV opened in the Dataset tab's Raw Data Explorer | `file`, `first_time` |
| `model_select` | a model card selected or deselected | `model_id`, `selected`, `selected_count` |
| `compare_view` | "Compare Selected" clicked | `models`, `model_count`, `includes_my_model`, `compare_index` |
| `compare_metric` | metric tab switched in Compare | `metric` |
| `template_download` | "Download template (my_model.csv)" clicked | — |
| `my_model_uploaded` | first page view after an accepted upload (once per upload) | `f1`, `auc`, `accuracy`, `precision`, `recall`, `runs`, `f1_band`, `elapsed_hr`, `gpu_hrs`, `gpu_tier_unlocked`, `task_complete`, `above_goal` |
| `milestone_gpu_tier_unlocked` | uploaded F1 above the best CPU model | `f1`, `threshold` |
| `milestone_task_complete` | uploaded F1 above the best GPU model | `f1`, `threshold` |
| `milestone_good_performance` | uploaded F1 above the 0.75 improvement goal | `f1`, `threshold` |
| `my_model_upload_failed` | upload rejected by validation | `problem_count`, `first_problem` |
| `my_model_upload_cleared` | "Clear upload" clicked | — |
| `ui_click` / `rage_click` | any button, link or label / 3 clicks within 0.8 s | `el_id`, `el_tag`, `el_text`, `el_action`, `link_url` |
| `ui_change` | slider or other control committed (e.g. the Reproducibility iteration slider) | `el_id`, `el_type`, `from_value`, `to_value`, `adjust_ms` |
| `scroll_depth` | 25/50/75/100% of the page or a scrolling table, once each per tab | `scroller`, `percent` |
| `text_copy` | student copies text from the page | `text_length` |
| `page_hidden` / `page_visible` | tab hidden / back | running totals `active_ms`, `tabs_seen`, `files_seen`, `compare_count`, `click_count` / `away_ms` |
| `js_error` | uncaught error (max 10 per page) | `message`, `source`, `line` |

An upload reloads the page, so its events are sent on the next page view. They are
sent once per upload: `analytics.js` stores the upload's timestamp under
`nnfp_ga_upload_reported`, so revisits don't count the same milestones again.

## Getting the data out (BigQuery)

GA4 reports show totals; per-student questions need the raw events. The GA4 property
already exports them daily to BigQuery (`nn-ui-analytics.analytics_557882182`, set up
for the assignment explorers). [`analysis/explorer_report.sql`](analysis/explorer_report.sql)
answers the usual questions from that export:

1. Time per user (visible and active minutes, visits)
2. Time per tab, per user
3. Tabs, class level: users per tab and median minutes
4. Dataset files opened
5. Comparisons: every comparison, then how often each model and metric was used
6. Every accepted upload with its scores, and the most common reasons uploads were rejected
7. Progress per user: best and latest F1, milestones reached, hours from first visit to each
8. Class funnel: viewed Deliverables → downloaded template → uploaded → GPU tier → task
   complete → above 0.75, with median and top F1
9. Score distribution in 0.05-wide F1 bands

Open BigQuery in the Cloud console, paste the file into a new query tab, set
`start_date` / `end_date` at the top, and Run. Each numbered SELECT returns its own
table: click "View results" next to it in the Results panel. Only final project events
(`assignment = final`) are counted; `?ga_debug=1` and localhost visits are left out.
The export runs once a day, so today's events show up tomorrow.

## Privacy

- No names, IDs or file names are sent. The uploaded `my_model.csv` file name could
  contain a student's name, so only its scores are sent. Google Analytics' terms forbid
  personally identifiable information.
- GA is anonymous: it shows how many browsers reached each milestone and the score
  distribution, not which student did. Grading still comes from D2L.
- The consent mode (`CONFIG.consent`, currently `'none'`) matches the assignment
  explorers. Analysing student interaction data for publication is human-subjects
  research. Check with the UArizona IRB before collecting, as described in the
  assignment explorers' ANALYTICS.md.
