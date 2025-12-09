### Check-list « v3.1 — SaaS hardening (ingestion ⇢ hypergraphe ⇢ coûts ⇢ serve) »

*(chaque ☐ = action à cocher ; indentation = sous-étapes ; formules + tableau des variables ; **Objectif** & **DoD** à la fin de chaque bloc)*

---

## 1 · Side-car pré-ingestion (safety + langue + gating audio/image) ★

* [x] **Déployer micro-service `ingest-filter` (Rust/Go)**
  ☑ Exposé via Envoy; intercepte avant Kafka.
  ☑ Retourne HTTP 422 si payload “fail”.
* [x] **Chaîne sécurité**

  1. Regex list 3 000 termes.
  2. Mini-model `distilroberta-toxic` → score $s_\text{tox}$.
  3. CLIP-NSFW → score $s_\text{nsfw}$.
  4. **Décision** block si

     $$
       s = \tfrac12(s_\text{tox}+s_\text{nsfw}) > 0.7
     $$
* [x] **LangID** fastText → skip BLIP/Whisper si $lang\notin\{\text{fr,en}\}$.
* [x] **Audio/Image gate** calculé avant push :

  $$
    \text{thr}_{SNR}
    = 6 + 0.5\,\sigma_{SNR},\quad
    \sigma_{SNR}=\sqrt{\tfrac1N\sum(SNR_i-\bar SNR)^2}
  $$
* [x] Métrique `snr_block_total` pour payloads audio en dessous du seuil SNR.
* **Objectif** : −10 % volume Kafka, taux faux positifs audio < 2 %.
* **DoD** : tests e2e → 95 % contenu tox bloqué, métriques `filter_block_total`, `lang_skipped_total` exposées.

---

## 2 · Hypergraphe “typed edges” + Laplacien multiplex ★

* [x] **Étendre schéma Neo4j** : labels `:EDGE_{type}` (`DOC`, `USER`, `TAG`).
* [x] **Laplacien 3-tenseur**

  $$
    \mathcal{L}^{(t)} = I - D_t^{-1/2}\,B_t\,W_t\,D_t^{-1}B_t^\top D_t^{-1/2}
  $$

  $$
    \Delta_\text{multi} = \sum_{t\in T} \alpha_t\;\mathcal{L}^{(t)},
    \quad \alpha_t \ge 0,\;\sum \alpha_t = 1
  $$

  | Var        | Signification                     |
  | ---------- | --------------------------------- |
  | $B_t$      | Incidence hyperarêtes type t      |
  | $W_t$      | Poids initiaux (=1)               |
  | $\alpha_t$ | poids apprenables (meta gradient) |
* [x] **Meta-gradient** — optimiser $\alpha_t$ sur Macro-F1 validation.
* **Objectif** : Macro-F1 communauté +1 pt.
* **DoD** : script d’apprentissage `train_alpha.py`; tests unit → contraintes $\sum \alpha=1$.

---

## 3 · Solver ILP « smart-patch » pour incohérences Sheaf ↔ Hyper ★☆

* [x] **Modèle** : variables binaires $x_e$ « appliquer patch sur arête ».

  $$
    \max_{x} \;\Delta S = \sum_e c_e x_e
    \quad\text{s.c.}\quad
    \sum_e w_e x_e \le B
  $$

  (poids = risque, $B$=budget patch).
* [x] Résolution via OR-Tools; stocke patchset id.
* [x] **Workflow UI** `/edge_review` : bouton *Apply*, *Reject*.
* **Objectif** : Cohérence S ↑ 10 % sans patch cassé.
* **DoD** : test propose patch cohérent ; rollback historique versionné.

---

## 4 · Flink watermark & late-event replay ★

* [x] `WatermarkStrategy.forBoundedOutOfOrderness(Duration.ofMinutes(10))`
  ☑ Side output `late-events` re-injected via Kafka topic `late_edges`.
* **Objectif** : Data loss < 0.1 %.
* **DoD** : integration test with 8 min delay → edge présent.

---

## 5 · GPU cost admission controller ★

* [x] **Mutating admission webhook** (`gpu-quota-webhook`)
  ☑ Calcule estimate $E_{gpu} = t_\text{epoch}\cdot n_\text{epoch}$.
  ☑ Refuse si `tenant_credits - E_gpu < 0`.
* [x] **Prometheus credits** :

  $$
    \text{credits_left} = credits_0 - \int gpu\_minutes\,dt
  $$
* **Objectif** : dépassement quota = 0.
* **DoD** : e2e submit job over-budget → HTTP 403.

---

## 6 · SmoothQuant bfloat4 + outlier clipping ★☆

* [x] **`training/quant_utils.py`** :
  ☑ Compute per-group scale $s_g = \max_{w\in g}|w|$/127.
  ☑ If $s_g > s_{P99}$ ⇒ clip.
* **DoD** : PPL ↑ < +2 %, CPU latency −5 %, aucune saturation.

---

## 7 · Drift threshold adaptatif par tenant ☆

* [x] **EWMA**

  $$
    \mu_{t}=\lambda d_t+(1-\lambda)\mu_{t-1},\;
    \sigma_{t} =\sqrt{\lambda(d_t-\mu_t)^2+(1-\lambda)\sigma_{t-1}^2}
  $$
* [x] Alerte si $d_t > \mu_t + 3\sigma_t$.
* **DoD** : faux positifs drift divisés par 5 sur tenants faibles.

---

## 8 · Metrics embedding CPU cost ☆

* [x] `analysis/embedding.py` : compteur Prom `embedding_cpu_seconds_total{tenant}`.
* **DoD** : billing dashboard affiche cost CPU + GPU.

---

## 9 · CronJob GC checkpoints (compléter) ☆

* [x] `k8s/cron/gc.yaml` : schedule `@daily`, job → `cleanup_checkpoints.py`.
* **DoD** : disque cluster < 80 %.

---

## 10 · Metrics requêtes Ray Serve ☆

* [x] `serving/ray_serve.py` : exposer `serve_requests_total{tenant,model_version}` et `serve_request_latency_seconds`.
* **DoD** : test unit → compteur + histogramme incrémentent.

---

## 11 · Metrics hypergraph late edges ☆

* [x] `hypergraph.py` : compteur Prom `late_edge_total` comptabilise les edges arrivées après le watermark.
* **DoD** : test unit → compteur incrémente quand un edge est routé vers `late_edges`.

---

## 12 · GPU quota submission metrics ☆

* [x] `gpu_quota_webhook.py` : compteur Prom `gpu_requests_total{tenant, status}`.
* **DoD** : test unit → compteur incrémente pour accept et reject.

---

## 13 · Metrics dérive tenant alerts ☆

* [x] `drift.py` : compteur Prom `drift_alert_total{tenant}` incrémenté à chaque alerte.
* **DoD** : test unit → compteur incrémente lors d'une alerte.

---

### KPI ciblés v3.1

| KPI                     | Cible    |
| ----------------------- | -------- |
| Kafka volume réduit     | –10 %    |
| Toxic false-negative    | ↓ ≥ 50 % |
| Macro-F1 commu.         | +1 pt    |
| Cohérence S après patch | +10 %    |
| Quota breach            | 0        |
| Data loss late event    | < 0.1 %  |

**Cochez tous les blocs pour passer Datacreek en v3.1-alpha SaaS compliant.**

### History
- Reset AGENTS for v3.1 checklist and imported tasks.
- Added embedding CPU seconds Prometheus metric with tests.
- Implemented per-tenant EWMA drift thresholds with alerting tests.
- Added daily checkpoint GC CronJob manifest and validation tests.
- Added SmoothQuant per-group scaling with P99 clipping and tests.
- Added Flink-style watermarking with late-event replay and tests.
- Added GPU quota admission webhook enforcing per-tenant credits with tests.
- Implemented multiplex Laplacian computation and alpha meta-gradient optimiser with tests.
- Added ILP smart-patch solver using OR-Tools with patchset registry and tests.
- Added FastAPI ingest-filter side-car with language gating, audio SNR threshold
  and toxic content checks, plus tests.
- Marked edge review workflow with apply/reject actions as complete.
- Extended Neo4j schema with typed edge labels and uniqueness constraints.
- Exposed per-tenant request counters and latency histograms for Ray Serve with tests.
- Added Prometheus counter for late hypergraph edges with tests.
- Added GPU quota submission counter with tests.
- Added drift alert Prometheus counter with tests.
- Added CPU cost tracker for billing dashboards with tests.
- Added SNR block metric for ingest filter to separate low SNR rejections.
- Re-ran linters and embedding CPU metric tests to verify implementation.
