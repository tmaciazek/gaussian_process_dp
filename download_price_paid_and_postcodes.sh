#!/usr/bin/env bash
# Download the input files expected by the property-price preprocessing code.
# Usage: bash download_price_paid_and_postcodes.sh [output-directory]

set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [output-directory]" >&2
    exit 2
fi

for command_name in curl unzip; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        exit 1
    fi
done

output_dir=${1:-.}
mkdir -p -- "$output_dir"
output_dir=$(cd -- "$output_dir" && pwd)
temporary_dir=$(mktemp -d "$output_dir/.download-data.XXXXXXXX")
trap 'rm -rf -- "$temporary_dir"' EXIT

download_csv() {
    local url=$1
    local destination=$2
    local temporary_file="$temporary_dir/${destination##*/}"

    if [[ -s "$destination" ]]; then
        echo "Already exists: $destination"
        return
    fi

    echo "Downloading $url"
    curl --fail --location --retry 3 --silent --show-error \
        --output "$temporary_file" "$url"
    if [[ ! -s "$temporary_file" ]]; then
        echo "Download was empty: $url" >&2
        exit 1
    fi
    mv -- "$temporary_file" "$destination"
    echo "Saved $destination"
}

# HM Land Registry yearly Price Paid Data (CSV files have no header row).
# https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
price_paid_base=https://price-paid-data.publicdata.landregistry.gov.uk
download_csv "$price_paid_base/pp-2017.csv" "$output_dir/pp-2017.csv"
download_csv "$price_paid_base/pp-2018.csv" "$output_dir/pp-2018.csv"

# The linked ONS item is ONSPD May 2026.
# Preserve the existing preprocessing filename NSPL.csv for its full UK CSV.
# https://geoportal.statistics.gov.uk/datasets/6fff67d204fd4f339591ed667a6e3642/about
ons_item=6fff67d204fd4f339591ed667a6e3642
ons_url="https://www.arcgis.com/sharing/rest/content/items/$ons_item/data"
ons_member=Data/ONSPD_MAY_2026_UK.csv

if [[ -s "$output_dir/NSPL.csv" ]]; then
    echo "Already exists: $output_dir/NSPL.csv"
else
    echo "Downloading ONS Postcode Directory (May 2026)"
    curl --fail --location --retry 3 --silent --show-error \
        --output "$temporary_dir/ons-postcodes.zip" "$ons_url"
    unzip -p "$temporary_dir/ons-postcodes.zip" "$ons_member" \
        > "$temporary_dir/NSPL.csv"
    if [[ ! -s "$temporary_dir/NSPL.csv" ]]; then
        echo "ONS archive did not contain a nonempty $ons_member" >&2
        exit 1
    fi
    mv -- "$temporary_dir/NSPL.csv" "$output_dir/NSPL.csv"
    echo "Saved $output_dir/NSPL.csv"
fi

echo "All input files are ready in $output_dir"
