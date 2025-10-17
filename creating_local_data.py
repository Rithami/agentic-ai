import requests
import csv
import time
import re

# ============== Configuration ===============
FDA_LABEL_API = "https://api.fda.gov/drug/label.json"
RXTERMS_INGREDIENT_API = "https://clinicaltables.nlm.nih.gov/api/drug_ingredients/v3/search"  

# ============== Helpers ===============

def fetch_fda_labels(limit=10, skip=0):
    """Fetch a batch of drug labels from FDA label API."""
    params = {"limit": limit, "skip": skip}
    resp = requests.get(FDA_LABEL_API, params=params)
    resp.raise_for_status()
    return resp.json().get("results", [])

def fetch_rxterms_ingredients(drug_name):
    """Use RxTerms API to get ingredients for a drug name."""
    params = {"drug": drug_name, "ef": "active_ingredient,inactive_ingredient"}
    resp = requests.get(RXTERMS_INGREDIENT_API, params=params)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if len(data) >= 3:
        details = data[2]
        return {
            "active": details.get("active_ingredient", []),
            "inactive": details.get("inactive_ingredient", [])
        }
    return None

def extract_complete_record(label):
    """Build a complete drug record using FDA or fallback to RxTerms API."""
    openfda = label.get("openfda", {})
    brand = openfda.get("brand_name", [""])[0].strip()
    generic = openfda.get("generic_name", [""])[0].strip()
    manufacturer = openfda.get("manufacturer_name", [""])[0].strip()
    app_no = openfda.get("application_number", [""])[0].strip()
    dosage_form = label.get("dosage_form", [""])[0].strip()
    active = label.get("active_ingredient", [])
    inactive = label.get("inactive_ingredient", [])

    # If both active & inactive exist in label, we have full data
    if active and inactive:
        return {
            "Brand Name": brand,
            "Generic Name": generic,
            "Manufacturer": manufacturer,
            "Application Number": app_no,
            "Dosage Form": dosage_form,
            "Active Ingredients": ", ".join(active),
            "Inactive Ingredients": ", ".join(inactive)
        }

    # Fallback to RxTerms API
    fallback = fetch_rxterms_ingredients(brand or generic)
    if fallback:
        active_f = fallback.get("active", [])
        inactive_f = fallback.get("inactive", [])
        if active_f and inactive_f:
            return {
                "Brand Name": brand or "Unknown brand",
                "Generic Name": generic or "Unknown generic",
                "Manufacturer": manufacturer or "Unknown",
                "Application Number": app_no or "Unknown",
                "Dosage Form": dosage_form or "Unknown",
                "Active Ingredients": ", ".join(active_f),
                "Inactive Ingredients": ", ".join(inactive_f)
            }

    return None

# ============== Collect 50 Complete Drugs ===============

def collect_50_complete_drugs():
    collected = []
    seen_apps = set()
    skip = 0
    batch_size = 10

    while len(collected) < 50:
        labels = fetch_fda_labels(limit=batch_size, skip=skip)
        if not labels:
            print("No more labels returned from FDA.")
            break

        for lab in labels:
            rec = extract_complete_record(lab)
            if rec:
                app = rec["Application Number"]
                if app not in seen_apps:
                    collected.append(rec)
                    seen_apps.add(app)
                    print(f"Collected {len(collected)}: {rec['Brand Name']} / {rec['Generic Name']}")
                    if len(collected) >= 50:
                        break
        skip += batch_size
        time.sleep(1)  # avoid API rate limits

    return collected

# ============== Normalize CSV Data ===============

def normalize_ingredients(records):
    """Split active/inactive ingredients into cleaned lists without duplicates."""
    for rec in records:
        # Clean and split active ingredients
        active_list = [i.strip() for i in rec["Active Ingredients"].split(",") if i.strip()]
        rec["Active Ingredients"] = ", ".join(list(dict.fromkeys(active_list)))

        # Clean and split inactive ingredients
        inactive_list = [i.strip() for i in rec["Inactive Ingredients"].split(",") if i.strip()]
        rec["Inactive Ingredients"] = ", ".join(list(dict.fromkeys(inactive_list)))

    return records

# ============== Main Execution ===============

if __name__ == "__main__":
    print("🔹 Collecting 50 complete drugs from FDA + RxTerms APIs...")
    records = collect_50_complete_drugs()
    
    if not records:
        print("❌ No complete records found.")
    else:
        print("🔹 Normalizing ingredients...")
        records = normalize_ingredients(records)

        # Write to CSV
        csv_file = "drug_ingredient.csv"
        keys = records[0].keys()
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"✅ Saved {len(records)} complete and normalized drug records to '{csv_file}'")