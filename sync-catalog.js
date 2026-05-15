/**
 * sync-catalog.js
 * Paginates through the JLL API for each city and builds jll-catalog.json.
 *
 * Usage:  node sync-catalog.js
 * Output: ../data/jll-catalog.json  (loaded by integration (1).js at startup)
 *
 * Run this once before starting the proxy, then re-run whenever the JLL
 * property inventory changes (e.g. weekly cron).
 */

require('dotenv').config();
const fs   = require('fs');
const path = require('path');

const JLL_BASE    = (process.env.JLL_BASE_URL || 'https://jll-backend.ibism.com').replace(/\/$/, '');
const OUTPUT_PATH = path.join(__dirname, '../data/jll-catalog.json');
const CITIES      = ['Chennai', 'Bengaluru', 'Hyderabad'];
const PAGE_DELAY  = 200;   // ms between pages (rate-limit courtesy)
const MAX_PAGES   = 500;   // safety cap per city (~10 000 properties)
const PAGE_SIZE   = 20;

// Ensure output directory exists
fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });

// ---------- helpers ----------------------------------------------------------

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function normalizeStr(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/[.\-_]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function fetchPage(city, page) {
  const qs = new URLSearchParams({ city, page: String(page), limit: String(PAGE_SIZE) });
  const url = `${JLL_BASE}/api/user/search/projects?${qs}`;
  const res = await fetch(url, { timeout: 15000 });
  if (!res.ok) throw new Error(`HTTP ${res.status} for page ${page}`);
  return res.json();
}

// ---------- main -------------------------------------------------------------

async function syncCity(city) {
  console.log(`\n[Sync] City: ${city}`);

  const locations     = new Set();
  const microMarkets  = new Set();
  const propertyTypes = new Set();
  const areaPrices    = {};   // areaName -> min price seen
  let   totalProps    = 0;
  let   cityMinPrice  = Infinity;
  let   cityMaxPrice  = 0;

  for (let pg = 1; pg <= MAX_PAGES; pg++) {
    let json;
    try {
      json = await fetchPage(city, pg);
    } catch (err) {
      console.warn(`[Sync] ${city} page ${pg} failed: ${err.message} — stopping`);
      break;
    }

    const items = Array.isArray(json?.data) ? json.data : [];
    if (items.length === 0) {
      console.log(`[Sync] ${city} page ${pg}: empty — done`);
      break;
    }

    totalProps += items.length;

    for (const p of items) {
      if (p.Location)    locations.add(p.Location);
      if (p.Micro_Market) microMarkets.add(p.Micro_Market);

      const pType = p.Project_Type || p.property_type || '';
      if (pType) propertyTypes.add(pType);

      // Compute per-area minimum price from configurations
      const area = p.Location || p.Micro_Market || '';
      const configs = p.configurations || p.configs || [];
      for (const c of configs) {
        const raw = Number(c.FinalPrice || c.All_Price || c.price || 0);
        if (!raw || raw <= 0) continue;
        // Normalise absurd values (stored in crore-units sometimes)
        const price = raw >= 1_000_000_000 ? raw / 1000 : raw;
        if (price > 0) {
          if (area) areaPrices[area] = Math.min(areaPrices[area] ?? Infinity, price);
          cityMinPrice = Math.min(cityMinPrice, price);
          cityMaxPrice = Math.max(cityMaxPrice, price);
        }
      }
    }

    process.stdout.write(`\r[Sync] ${city} — page ${pg}, props so far: ${totalProps}   `);
    await sleep(PAGE_DELAY);
  }

  console.log(`\n[Sync] ${city} done: ${totalProps} properties, ${locations.size} locations, ${microMarkets.size} micro-markets`);

  return {
    locations:      [...locations].sort(),
    micro_markets:  [...microMarkets].sort(),
    property_types: [...propertyTypes].sort(),
    area_min_prices: areaPrices,
    total_properties: totalProps,
    min_price: cityMinPrice === Infinity ? null : cityMinPrice,
    max_price: cityMaxPrice === 0        ? null : cityMaxPrice,
  };
}

async function main() {
  console.log(`[Sync] JLL Catalog Sync starting — output: ${OUTPUT_PATH}`);
  console.log(`[Sync] Cities: ${CITIES.join(', ')}`);

  const byCity        = {};
  const allTypes      = new Set();

  for (const city of CITIES) {
    const data = await syncCity(city);
    byCity[city] = data;
    data.property_types.forEach(t => allTypes.add(t));
  }

  const catalog = {
    synced_at:      new Date().toISOString(),
    cities:         CITIES,
    by_city:        byCity,
    property_types: [...allTypes].sort(),
  };

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(catalog, null, 2), 'utf8');
  console.log(`\n[Sync] Catalog written to ${OUTPUT_PATH}`);
  console.log(`[Sync] Summary:`);
  for (const city of CITIES) {
    const d = byCity[city];
    console.log(`  ${city}: ${d.total_properties} props, ${d.locations.length} locations, types: [${d.property_types.join(', ')}]`);
  }

  // Hot-reload: tell the running proxy to reload the catalog
  try {
    const res = await fetch('http://localhost:3000/api/integration/catalog/reload', { method: 'POST', timeout: 3000 });
    if (res.ok) console.log('[Sync] Hot-reload: proxy reloaded catalog');
  } catch (_) {
    console.log('[Sync] Hot-reload skipped (proxy not running — start proxy then search will use new catalog)');
  }
}

main().catch(err => { console.error('[Sync] Fatal:', err); process.exit(1); });
