"""
mlops/airflow/dags/segmentation_pipeline_dag.py
─────────────────────────────────────────────────
Weekly automated pipeline DAG for geo-behavioral segmentation refresh.

Schedule: Every Sunday at 03:00 UTC
Pipeline:
  1. data_validation     — validate new subscriber data quality
  2. feature_engineering — rebuild feature matrix
  3. clustering          — re-run KMeans segmentation
  4. segment_profiling   — auto-name and describe segments
  5. recommender_train   — refresh recommendation engine
  6. geo_density         — update H3 segment density map
  7. drift_detection     — check for significant segment drift
  8. conditional_promote — promote model to Production if quality threshold met
  9. notify              — send summary to Slack/email
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner":           "data-science-team",
    "depends_on_past": False,
    "start_date":      days_ago(1),
    "email_on_failure":True,
    "retries":         2,
    "retry_delay":     timedelta(minutes=5),
    "execution_timeout": timedelta(hours=3),
}

dag = DAG(
    dag_id="geo_customer_segmentation",
    default_args=default_args,
    description="Weekly geo-behavioral customer segmentation refresh",
    schedule_interval="0 3 * * 0",  # Sunday 03:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["segmentation", "geospatial", "recommender", "production"],
)


# ── Task functions ─────────────────────────────────────────────────────────────

def run_data_validation(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/data_engineering/data_validation.py",
         "--input", "data/raw/subscribers.parquet"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Validation failed:\n{r.stderr}")
    ctx["task_instance"].xcom_push(key="validation_passed", value=True)


def run_feature_engineering(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/features/feature_pipeline.py"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Feature engineering failed:\n{r.stderr}")


def run_clustering(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/models/clustering.py", "--algorithm", "kmeans"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Clustering failed:\n{r.stderr}")

    # Extract silhouette from stdout for downstream tasks
    for line in r.stdout.split("\n"):
        if "Silhouette" in line:
            try:
                sil = float(line.split(":")[-1].strip())
                ctx["task_instance"].xcom_push(key="silhouette", value=sil)
            except ValueError:
                pass


def run_segment_profiling(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/models/segment_profiler.py"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Profiling failed:\n{r.stderr}")


def run_recommender_train(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/models/recommender.py", "--train"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Recommender training failed:\n{r.stderr}")


def run_geo_density(**ctx):
    import subprocess
    r = subprocess.run(
        ["python", "src/models/geo_density.py"],
        capture_output=True, text=True, cwd="/app"
    )
    if r.returncode != 0:
        raise ValueError(f"Geo density failed:\n{r.stderr}")


def check_segment_drift(**ctx):
    """
    Compare current segment distribution to the previous run.
    If >15% of subscribers changed segments, trigger re-profiling.
    Returns branch name.
    """
    import pandas as pd
    import yaml

    config = yaml.safe_load(open("configs/config.yaml"))
    threshold = config["monitoring"]["drift_threshold_pct"]

    current_path  = Path("data/processed/segment_labels.parquet")
    previous_path = Path("data/processed/segment_labels_previous.parquet")

    if not previous_path.exists():
        # First run — save and skip drift check
        if current_path.exists():
            import shutil
            shutil.copy(current_path, previous_path)
        ctx["task_instance"].xcom_push(key="drift_pct", value=0.0)
        return "skip_rerun"

    curr = pd.read_parquet(current_path)
    prev = pd.read_parquet(previous_path)
    merged = curr.merge(prev, on="subscriber_id", suffixes=("_new", "_old"))
    drift_pct = (merged["segment_id_new"] != merged["segment_id_old"]).mean() * 100

    ctx["task_instance"].xcom_push(key="drift_pct", value=round(drift_pct, 2))

    # Save current as previous for next run
    import shutil
    shutil.copy(current_path, previous_path)

    if drift_pct > threshold:
        return "high_drift_alert"
    return "skip_rerun"


def send_high_drift_alert(**ctx):
    """Log a drift alert (hook into Slack/email in production)."""
    drift_pct = ctx["task_instance"].xcom_pull(
        task_ids="drift_detection", key="drift_pct"
    ) or 0.0
    message = (
        f"⚠️  HIGH SEGMENT DRIFT DETECTED\n"
        f"   {drift_pct:.1f}% of subscribers changed segments this week.\n"
        f"   Consider reviewing clustering parameters or re-profiling.\n"
    )
    print(message)
    # In production: post to Slack, send email, create Jira ticket, etc.


def promote_model_if_good(**ctx):
    """Promote model to Production in MLflow if silhouette >= threshold."""
    import yaml
    config = yaml.safe_load(open("configs/config.yaml"))
    min_sil = config["monitoring"]["min_silhouette_score"]
    sil = ctx["task_instance"].xcom_pull(
        task_ids="clustering", key="silhouette"
    ) or 0.0

    if sil >= min_sil:
        try:
            import mlflow
            mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(
                config["mlflow"]["registered_model_name"], stages=["None", "Staging"]
            )
            if versions:
                latest = sorted(versions, key=lambda v: int(v.version))[-1]
                client.transition_model_version_stage(
                    name=config["mlflow"]["registered_model_name"],
                    version=latest.version,
                    stage="Production",
                )
                print(f"✅ Model v{latest.version} promoted to Production (sil={sil:.3f})")
        except Exception as e:
            print(f"MLflow promotion failed: {e}")
    else:
        print(f"⚠️  Model NOT promoted (sil={sil:.3f} < threshold {min_sil})")


def send_pipeline_notification(**ctx):
    """Send weekly pipeline summary."""
    ti = ctx["task_instance"]
    sil       = ti.xcom_pull(task_ids="clustering",      key="silhouette") or 0.0
    drift_pct = ti.xcom_pull(task_ids="drift_detection", key="drift_pct") or 0.0
    run_date  = ctx["ds"]

    msg = (
        f"📍 Segmentation Pipeline — {run_date}\n"
        f"  Silhouette score : {sil:.4f}\n"
        f"  Segment drift    : {drift_pct:.1f}%\n"
    )
    print(msg)
    # Uncomment to push to Slack:
    # import requests
    # requests.post(Variable.get("SLACK_WEBHOOK"), json={"text": msg})


# ── Task objects ───────────────────────────────────────────────────────────────
t_start = EmptyOperator(task_id="pipeline_start", dag=dag)

t_validate = PythonOperator(task_id="data_validation",
                             python_callable=run_data_validation, dag=dag)

t_features = PythonOperator(task_id="feature_engineering",
                             python_callable=run_feature_engineering, dag=dag)

t_cluster  = PythonOperator(task_id="clustering",
                             python_callable=run_clustering, dag=dag)

t_profile  = PythonOperator(task_id="segment_profiling",
                             python_callable=run_segment_profiling, dag=dag)

t_rec      = PythonOperator(task_id="recommender_training",
                             python_callable=run_recommender_train, dag=dag)

t_geo      = PythonOperator(task_id="geo_density_refresh",
                             python_callable=run_geo_density, dag=dag)

t_drift    = BranchPythonOperator(task_id="drift_detection",
                                   python_callable=check_segment_drift, dag=dag)

t_alert    = PythonOperator(task_id="high_drift_alert",
                             python_callable=send_high_drift_alert, dag=dag)

t_skip     = EmptyOperator(task_id="skip_rerun", dag=dag)

t_promote  = PythonOperator(task_id="promote_model",
                             python_callable=promote_model_if_good,
                             trigger_rule="none_failed_min_one_success", dag=dag)

t_notify   = PythonOperator(task_id="send_notification",
                             python_callable=send_pipeline_notification,
                             trigger_rule="none_failed_min_one_success", dag=dag)

t_end = EmptyOperator(task_id="pipeline_end", dag=dag)

# ── DAG wiring ─────────────────────────────────────────────────────────────────
# start → validate → features → cluster → profile → recommender → geo → drift?
#                                                                       ├─ alert ─┐
#                                                                       └─ skip  ─┤
#                                                                                  └→ promote → notify → end
(t_start >> t_validate >> t_features >> t_cluster >> t_profile
 >> t_rec >> t_geo >> t_drift)
t_drift  >> t_alert >> t_promote
t_drift  >> t_skip  >> t_promote
t_promote >> t_notify >> t_end
