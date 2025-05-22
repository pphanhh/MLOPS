# pipeline_grafana.py
import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Grafana configuration
grafana_host    = os.getenv("GRAFANA_HOST", "http://localhost:3000")
grafana_api_key = os.getenv("GRAFANA_API_KEY")  # Service Account token or API key
# Optional: Org ID if not default
grafana_org_id  = os.getenv("GRAFANA_ORG_ID", "1")
headers = {
    "Authorization": f"Bearer {grafana_api_key}",
    "Content-Type":  "application/json"
}

# PostgreSQL datasource settings
datasource_name = os.getenv("DATASOURCE_NAME", "PostgreSQL - Twitter Analysis")
postgres_host   = os.getenv("DB_HOST", "localhost")
postgres_port   = os.getenv("DB_PORT", "5432")
postgres_db     = os.getenv("DB_NAME", "twitter_analysis")
postgres_user   = os.getenv("DB_USER", "postgres")
postgres_pass   = os.getenv("DB_PASSWORD", "")

# Dashboard cloning settings
orig_dashboard_uid = os.getenv("ORIG_DASHBOARD_UID")  # e.g. "eeksft5ash9tsf"
mloops_dashboard_uid = os.getenv("MLOPS_DASHBOARD_UID", "mlops")


def upsert_datasource():
    """Create or update the Postgres datasource in Grafana."""
    payload = {
        "name": datasource_name,
        "type": "postgres",
        "access": "proxy",
        "url": f"{postgres_host}:{postgres_port}",
        "database": postgres_db,
        "user": postgres_user,
        "jsonData": {"sslmode": "disable", "postgresVersion": 1300},
        "secureJsonData": {"password": postgres_pass}
    }
    # Check existing
    get_url = f"{grafana_host}/api/datasources/name/{datasource_name}"
    r = requests.get(get_url, headers=headers)
    if r.status_code == 200:
        ds = r.json()
        payload["id"] = ds["id"]
        # Update
        put_url = f"{grafana_host}/api/datasources/{ds['id']}"
        r2 = requests.put(put_url, json=payload, headers=headers)
        r2.raise_for_status()
        print(f"✅ Updated datasource '{datasource_name}' (id={ds['id']})")
    else:
        # Create new
        post_url = f"{grafana_host}/api/datasources"
        r3 = requests.post(post_url, json=payload, headers=headers)
        r3.raise_for_status()
        print(f"✅ Created datasource '{datasource_name}'")


def clone_dashboard():
    """
    Clone a manually created dashboard to a new dashboard UID (mlops).
    Requires ORIG_DASHBOARD_UID in .env (UID from the dashboard URL you designed).
    Optionally set GRAFANA_ORG_ID if using a non-default org.
    """
    if not orig_dashboard_uid:
        raise ValueError("ORIG_DASHBOARD_UID not set in .env")

    # Fetch original dashboard JSON model
    url_get = f"{grafana_host}/api/dashboards/uid/{orig_dashboard_uid}?orgId={grafana_org_id}"
    r = requests.get(url_get, headers=headers)
    r.raise_for_status()
    model = r.json()["dashboard"]

    # Adjust metadata for MLOPS dashboard
    model["uid"] = mloops_dashboard_uid
    model["id"] = None
    model["version"] = 0
    model["title"] = os.getenv("MLOPS_DASHBOARD_TITLE", model.get("title", "MLOPS"))

    payload = {"dashboard": model, "overwrite": True, "folderId": 0}
    # Deploy cloned dashboard
    post_url = f"{grafana_host}/api/dashboards/db"
    resp = requests.post(post_url, json=payload, headers=headers)
    resp.raise_for_status()
    print(f"✅ Cloned dashboard '{orig_dashboard_uid}' to '{mloops_dashboard_uid}'.")


def main():
    upsert_datasource()
    clone_dashboard()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
