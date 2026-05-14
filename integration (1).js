/**
 * Integration proxy routes
 *
 * Proxies JLL API calls from the voice agent, performs location filtering with
 * alias expansion and punctuation-insensitive matching, and logs all tool calls
 * to integration_requests for audit/debugging.
 *
 * Location strategy:
 *   - Forward `location` to JLL (their API does server-side filtering when the
 *     value matches their stored format, e.g. "T. Nagar").
 *   - ALWAYS also apply client-side filter as a guard, because JLL field values
 *     use dotted abbreviations ("T. Nagar") that don't match plain spoken input
 *     ("T Nagar"). Normalization strips punctuation before comparing.
 *   - If JLL returns 0 results (wrong format, unlisted area), auto-retry without
 *     location so we get the full city set to filter client-side.
 *
 * No auth required â€” internal service call from voice agent only.
 */

const express = require('express');
const router  = express.Router();
// Stub pool — no database in standalone mode; audit queries become no-ops
const pool = { query: async () => ({ rows: [] }) };
// Node v24+ has built-in global fetch — no need for node-fetch
const fs      = require('fs');
const path    = require('path');

const JLL_BASE = process.env.JLL_API_URL || 'https://jll-backend.ibism.com';

// â”€â”€â”€ In-memory search result cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Caches processed search responses (after filtering + accumulation) keyed by
// a normalized hash of the search parameters.
// TTL: 3 minutes â€” fresh enough for a call, short enough to avoid stale data.
// This avoids repeat JLL API round-trips for the same search params within a
// single call session (e.g. user says "show more" twice for the same criteria).
const _searchCache   = new Map();  // key â†’ { data, ts }
const CACHE_TTL_MS   = 180_000;   // 3 minutes

// â”€â”€â”€ In-memory areas-by-budget cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// areas_by_budget scans up to 15 JLL pages sequentially â€” very expensive.
// Cache the full area list for a given city+type+budget for 5 minutes so
// repeated calls (same call, different pages, or back-to-back calls) are instant.
const _areasCache       = new Map();  // key â†’ { areas, ts }
const AREAS_CACHE_TTL   = 300_000;   // 5 minutes
const _propertyResolveCache = new Map(); // key -> { data, ts }
const PROPERTY_RESOLVE_TTL_MS = 300_000;
const _propertyIndexCache = new Map(); // city -> { items, ts }
const PROPERTY_INDEX_TTL_MS = 1_800_000; // 30 minutes
const PROPERTY_INDEX_MAX_PAGES = 30;
const PROPERTY_INDEX_CONCURRENCY = 5;

function _areasCacheKey(city, type, minPrice, maxPrice) {
  return `${(city || '').toLowerCase()}|${(type || '').toLowerCase()}|${minPrice || 0}|${maxPrice || 'inf'}`;
}
function _getAreasCache(key) {
  const hit = _areasCache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > AREAS_CACHE_TTL) { _areasCache.delete(key); return null; }
  return hit.areas;
}
function _setAreasCache(key, areas) {
  if (_areasCache.size > 100) _areasCache.delete(_areasCache.keys().next().value);
  _areasCache.set(key, { areas, ts: Date.now() });
}

function _cacheKey(params) {
  const { city, location, property_type, project_category, construction_state, min_price, max_price, page, sort_preference, bedrooms } = params;
  return JSON.stringify({
    city: (city || '').toLowerCase().trim(),
    location: (location || '').toLowerCase().trim(),
    property_type: (property_type || '').toLowerCase().trim(),
    bedrooms: (bedrooms || '').toLowerCase().trim(),
    project_category: (project_category || '').toLowerCase().trim(),
    construction_state: (construction_state || '').toLowerCase().trim(),
    min_price: min_price || '',
    max_price: max_price || '',
    page: page || '1',
    sort: sort_preference || '',
  });
}

function _getCached(key) {
  const hit = _searchCache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > CACHE_TTL_MS) {
    _searchCache.delete(key);
    return null;
  }
  return hit.data;
}

function _propertyDisplayName(p) {
  return p?.project_name || p?.name || p?.Project_Name_Original || p?.Project_Name || p?.Project_Name_New || '?';
}

function _propertySummary(p) {
  const name = _propertyDisplayName(p);
  const type = p?.config_summary || p?.type || '-';
  const location = p?.location || p?.micro_market || p?.city || '-';
  const price = p?.starting_price || '-';
  const category = p?.category || p?.Project_Category || '-';
  const construction = p?.construction || p?.State_Of_Construction || '-';
  return `${name} (${type}) @ ${location} @ ${price} @ category=${category} @ construction=${construction}`;
}

function _logBackendProperties(prefix, properties) {
  const items = Array.isArray(properties) ? properties : [];
  if (!items.length) {
    console.log(`[BackendProperties] ${prefix}: 0 properties`);
    return;
  }
  const summaries = items.map((item, index) => `  ${index + 1}. ${_propertySummary(item)}`).join('\n');
  console.log(`[BackendProperties] ${prefix}: ${items.length} properties\n${summaries}`);
}

function _setCache(key, data) {
  // Evict oldest entries when cache grows large (>200 entries)
  if (_searchCache.size > 200) {
    const oldestKey = _searchCache.keys().next().value;
    _searchCache.delete(oldestKey);
  }
  _searchCache.set(key, { data, ts: Date.now() });
}

function _getTimedCache(map, key, ttlMs) {
  const hit = map.get(key);
  if (!hit) return null;
  if (Date.now() - hit.ts > ttlMs) {
    map.delete(key);
    return null;
  }
  return hit.data;
}

function _setTimedCache(map, key, data, maxSize = 200) {
  if (map.size > maxSize) {
    map.delete(map.keys().next().value);
  }
  map.set(key, { data, ts: Date.now() });
}

// â”€â”€â”€ Location alias map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// JLL stores micro-market names ("Kelambakkam", "Sholinganallur") rather than
// corridor names users commonly say ("OMR", "ECR"). This map expands a spoken
// location into all the suburb names that fall within that corridor so the
// client-side filter can match them.
const LOCATION_ALIASES = {
  // Chennai corridors
  'omr':          ['omr', 'old mahabalipuram', 'sholinganallur', 'perumbakkam', 'kelambakkam',
                   'siruseri', 'navallur', 'navalur', 'karapakkam', 'thoraipakkam',
                   'thaiyur', 'shollinganallur', 'perungudi', 'egattur', 'medavakkam'],
  'ecr':          ['ecr', 'east coast road', 'kottivakkam', 'injambakkam', 'akkarai',
                   'uthandi', 'neelankarai', 'kovalam', 'marakkanam', 'palavakkam',
                   'vettuvankeni', 'kanathur'],
  'gsth':         ['gsth', 'grand southern trunk', 'tambaram', 'chromepet', 'pallavaram',
                   'perungalathur', 'vandalur', 'guduvanchery', 'urapakkam'],
  'anna nagar':   ['anna nagar', 'annanagar', 'anna nagar west', 'anna nagar east'],
  't nagar':      ['t nagar', 't. nagar', 'tnagar', 't.nagar', 'teynampet'],
  // Bengaluru corridors
  'sarjapur':     ['sarjapur', 'sarjapura', 'sarjapur road', 'attibele'],
  'whitefield':   ['whitefield', 'white field', 'kadugodi', 'hoodi', 'varthur'],
  'electronic city': ['electronic city', 'electronics city', 'hosa road', 'begur'],
  'hebbal':       ['hebbal', 'yelahanka', 'devanahalli', 'bagalur', 'kogilu'],
  // Hyderabad
  'hitech city':  ['hitech city', 'hi-tech city', 'hitec city', 'madhapur', 'kondapur',
                   'gachibowli', 'nanakramguda', 'raidurgam'],
  'kokapet':      ['kokapet', 'narsingi', 'puppalaguda', 'financial district'],
};

/**
 * Normalize a location string for fuzzy matching:
 * strips dots, hyphens, extra whitespace â†’ "T. Nagar" â†’ "t nagar"
 * This is applied to BOTH the search token AND the JLL field values before
 * comparing, so "T. Nagar" (JLL stored format) matches spoken "T Nagar".
 */
function normalizeStr(s) {
  return (s || '').toLowerCase()
    .replace(/\./g, ' ')       // T. Nagar  â†’ T  Nagar
    .replace(/-/g, ' ')        // Hi-Tech    â†’ Hi Tech
    .replace(/\s+/g, ' ')      // T  Nagar   â†’ T Nagar
    .trim();
}

/**
 * Format an INR price value into a human-readable string.
 * e.g. 4500000 â†’ "45 L",  12500000 â†’ "1.3 Cr"
 * Returns null if val is falsy or 0.
 */
function normalizeProjectPriceINR(val) {
  let n = Number(val);
  if (!n || !isFinite(n) || n <= 0) return null;
  while (n >= 1_000_000_000) {
    n = n / 1000;
  }
  return Math.round(n);
}

function formatPriceINR(val) {
  const n = normalizeProjectPriceINR(val);
  if (!n || !isFinite(n) || n <= 0) return null;
  if (n >= 10_000_000) {
    let crore = Math.floor(n / 10_000_000);
    let lakh = Math.round((n % 10_000_000) / 100_000);
    if (lakh >= 100) {
      crore += 1;
      lakh = 0;
    }
    return lakh > 0 ? `${crore} Crore ${lakh} Lakh` : `${crore} Crore`;
  }
  if (n >= 100_000)    return `${Math.round(n / 100_000)} Lakh`;
  return `â‚¹${n.toLocaleString('en-IN')}`;
}

function _normalizePropertyNameForMatch(value) {
  return normalizeStr(String(value || ''))
    .replace(/\b(apartment|apartments|flat|flats|villa|villas|plot|plots|property|project|bhk)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function _parsePropertyResolveVariants(raw) {
  const values = [];
  const add = (value) => {
    const text = String(value || '').trim();
    if (text) values.push(text);
  };

  if (Array.isArray(raw)) {
    raw.forEach(add);
  } else if (typeof raw === 'string' && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) parsed.forEach(add);
      else add(parsed);
    } catch {
      raw.split(/[|,]/).forEach(add);
    }
  } else if (raw) {
    add(raw);
  }

  const seen = new Set();
  const deduped = [];
  for (const value of values) {
    const key = _normalizePropertyNameForMatch(value);
    if (!key || seen.has(key)) continue;
    seen.add(key);
    deduped.push(value);
    if (deduped.length >= 8) break;
  }
  return deduped;
}

function _propertyResolveQueries(name, variants = []) {
  const seen = new Set();
  const queries = [];
  for (const value of [name, ...variants]) {
    const text = String(value || '').trim();
    const key = _normalizePropertyNameForMatch(text);
    if (!text || !key || seen.has(key)) continue;
    seen.add(key);
    queries.push(text);
    if (queries.length >= 9) break;
  }
  return queries;
}

function _scorePropertyCandidateForQueries(candidate, queries, locationHint = '') {
  let bestScore = -1;
  let bestQuery = '';
  for (const query of queries) {
    const score = _scorePropertyCandidate(candidate, query, locationHint);
    if (score > bestScore) {
      bestScore = score;
      bestQuery = query;
    }
  }
  return { score: bestScore, matchedQuery: bestQuery };
}

function _propertyResolveCacheKey({ city, name, location_hint, variants = [] }) {
  return JSON.stringify({
    city: normalizeStr(city || ''),
    name: _normalizePropertyNameForMatch(name || ''),
    location_hint: normalizeStr(location_hint || ''),
    variants: variants.map(v => _normalizePropertyNameForMatch(v)).filter(Boolean).sort(),
  });
}

function _projectNameTokens(value) {
  return _normalizePropertyNameForMatch(value)
    .split(' ')
    .map(t => t.trim())
    .filter(t => t.length >= 3);
}

function _levenshteinDistance(a, b) {
  const left = String(a || '');
  const right = String(b || '');
  if (!left) return right.length;
  if (!right) return left.length;
  const dp = Array.from({ length: right.length + 1 }, (_, idx) => idx);
  for (let i = 1; i <= left.length; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= right.length; j++) {
      const temp = dp[j];
      if (left[i - 1] === right[j - 1]) {
        dp[j] = prev;
      } else {
        dp[j] = Math.min(prev + 1, dp[j] + 1, dp[j - 1] + 1);
      }
      prev = temp;
    }
  }
  return dp[right.length];
}

function _normalizedEditSimilarity(a, b) {
  const left = _normalizePropertyNameForMatch(a);
  const right = _normalizePropertyNameForMatch(b);
  if (!left || !right) return 0;
  const maxLen = Math.max(left.length, right.length);
  if (!maxLen) return 0;
  return 1 - (_levenshteinDistance(left, right) / maxLen);
}

function _summarizeConfigs(configs) {
  if (!Array.isArray(configs) || configs.length === 0) return null;
  const types = [...new Set(configs.map(c => c.Config_Type || c.type).filter(Boolean))];
  if (!types.length) return null;
  const nums = types.map(t => (String(t).match(/^\d+/) || [])[0]).filter(Boolean);
  if (nums.length > 0 && types.every(t => /BHK/i.test(String(t)))) {
    return `${nums.join('/')} BHK`;
  }
  return types.slice(0, 3).join('/');
}

function _normalizeProjectItemForResolve(p) {
  const configs = p.configurations || p.configs || [];
  const prices = configs
    .map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || c.price || 0))
    .filter(n => n > 0);
  const minPriceVal = prices.length ? Math.min(...prices) : null;
  return {
    property_id: p.Project_Slug || p.project_slug || p.property_id || null,
    project_slug: p.Project_Slug || p.project_slug || p.property_id || null,
    name: p.Project_Name_Original || p.Project_Name || p.Project_Name_New || p.name || null,
    full_name: p.Project_Name || p.full_name || null,
    seo_name: p.Project_Name_New || p.seo_name || null,
    location: p.Location || p.location || null,
    micro_market: p.Micro_Market || p.micro_market || null,
    city: p.City || p.city || null,
    bhk: _summarizeConfigs(configs),
    price: minPriceVal ? formatPriceINR(minPriceVal) : null,
  };
}

function _candidateDisplayBits(candidate) {
  const area = candidate.location || candidate.micro_market || candidate.city || 'unknown area';
  const bhk = candidate.bhk || 'property';
  const price = candidate.price || 'price on request';
  return `${candidate.name} (${bhk}) @ ${area} @ ${price}`;
}

function _firstNonEmptyValue(...values) {
  for (const value of values) {
    if (value == null) continue;
    if (typeof value === 'boolean') {
      if (value) return 'RERA approved';
      continue;
    }
    if (Array.isArray(value)) {
      const nested = _firstNonEmptyValue(...value);
      if (nested) return nested;
      continue;
    }
    if (typeof value === 'object') {
      const nested = _firstNonEmptyValue(
        value.RERA_No,
        value.RERA_Number,
        value.RERA_Registration_No,
        value.Rera_Number,
        value.ReraNo,
        value.reraNo,
        value.registrationNo,
        value.registration_number,
        value.reraStatus,
        value.rera_status,
        value.rera_approval_status,
        value.reraApprovalStatus,
        value.reraApproved,
        value.isReraApproved,
        value.is_rera_approved,
        value.number,
        value.value,
        value.status
      );
      if (nested) return nested;
      continue;
    }
    const text = String(value).trim();
    if (text) return text;
  }
  return null;
}

function _extractProjectRera(p) {
  return _firstNonEmptyValue(
    p.RERA_No,
    p.RERA_Number,
    p.RERA_Registration_No,
    p.RERA_Registration_Number,
    p.RERA_Number_List,
    p.Rera_Number,
    p.ReraNo,
    p.Rera_No,
    p.reraNo,
    p.rera_number,
    p.reraRegistrationNo,
    p.RERA,
    p.Rera,
    p.rera,
    p.reraDetails,
    p.rera_details,
    p.reraStatus,
    p.rera_status,
    p.rera_approval_status,
    p.reraApprovalStatus,
    p.reraApproved,
    p.isReraApproved,
    p.is_rera_approved,
    p.approvals && p.approvals.rera
  );
}

function _scorePropertyCandidate(candidate, queryName, locationHint = '') {
  const names = [
    candidate.name,
    candidate.full_name,
    candidate.seo_name,
  ].filter(Boolean);
  const normQuery = _normalizePropertyNameForMatch(queryName);
  const queryTokens = _projectNameTokens(queryName);
  if (!normQuery || !queryTokens.length) return -1;

  let best = -1;
  for (const rawName of names) {
    const normName = _normalizePropertyNameForMatch(rawName);
    if (!normName) continue;
    let score = 0;
    if (normName === normQuery) score += 100;
    else if (normName.startsWith(normQuery) || normQuery.startsWith(normName)) score += 88;
    else if (normName.includes(normQuery) || normQuery.includes(normName)) score += 80;

    const nameTokens = new Set(_projectNameTokens(rawName));
    const matchedTokens = queryTokens.filter(token => nameTokens.has(token));
    score += matchedTokens.length * 16;
    if (matchedTokens.length === queryTokens.length) score += 18;
    if (matchedTokens.length === 1 && queryTokens.length > 1) score -= 12;

    const wholeSimilarity = _normalizedEditSimilarity(rawName, queryName);
    if (wholeSimilarity >= 0.88) score += 34;
    else if (wholeSimilarity >= 0.8) score += 24;
    else if (wholeSimilarity >= 0.72) score += 14;

    if (queryTokens.length && nameTokens.size) {
      let fuzzyTokenHits = 0;
      for (const qToken of queryTokens) {
        for (const nToken of nameTokens) {
          if (qToken === nToken) continue;
          if (_normalizedEditSimilarity(qToken, nToken) >= 0.8) {
            fuzzyTokenHits += 1;
            break;
          }
        }
      }
      score += fuzzyTokenHits * 6;
    }

    if (locationHint) {
      const hint = normalizeStr(locationHint);
      const hay = normalizeStr([
        candidate.location || '',
        candidate.micro_market || '',
        candidate.city || '',
      ].join(' '));
      if (hint && hay.includes(hint)) score += 14;
    }

    if (candidate.bhk) score += 1;
    if (candidate.price) score += 1;
    best = Math.max(best, score);
  }
  return best;
}

async function _fetchProjectIndexPage(city, pageNum, locationHint = '') {
  const qs = new URLSearchParams({
    city: city || '',
    page: String(pageNum),
    limit: '20',
  });
  if (locationHint) qs.set('location', locationHint);
  const resp = await fetch(`${JLL_BASE}/api/user/search/projects?${qs}`, { timeout: 10000 });
  const json = await resp.json();
  const items = Array.isArray(json?.data) ? json.data : [];
  const total = Number(json?.total ?? json?.totalCount ?? json?.total_count ?? 0) || 0;
  return { items, total };
}

async function _buildPropertyIndex(city, locationHint = '') {
  const first = await _fetchProjectIndexPage(city, 1, locationHint);
  const totalPages = Math.max(1, Math.ceil((first.total || first.items.length || 0) / 20));
  const pageLimit = Math.min(PROPERTY_INDEX_MAX_PAGES, totalPages || PROPERTY_INDEX_MAX_PAGES);
  const allRaw = [...first.items];

  for (let start = 2; start <= pageLimit; start += PROPERTY_INDEX_CONCURRENCY) {
    const batchPages = [];
    for (let pg = start; pg < start + PROPERTY_INDEX_CONCURRENCY && pg <= pageLimit; pg++) {
      batchPages.push(pg);
    }
    const batchResults = await Promise.all(
      batchPages.map(pg => _fetchProjectIndexPage(city, pg, locationHint).catch(() => ({ items: [], total: 0 })))
    );
    for (const result of batchResults) {
      if (Array.isArray(result.items) && result.items.length) {
        allRaw.push(...result.items);
      }
    }
  }

  const deduped = new Map();
  for (const raw of allRaw) {
    const normalized = _normalizeProjectItemForResolve(raw);
    if (!normalized.property_id || !normalized.name) continue;
    if (!deduped.has(normalized.property_id)) {
      deduped.set(normalized.property_id, normalized);
    }
  }
  return Array.from(deduped.values());
}

async function _getPropertyIndex(city) {
  const cacheKey = normalizeStr(city || 'Chennai') || 'chennai';
  const hit = _getTimedCache(_propertyIndexCache, cacheKey, PROPERTY_INDEX_TTL_MS);
  if (hit) return hit;
  const items = await _buildPropertyIndex(city || 'Chennai');
  _setTimedCache(_propertyIndexCache, cacheKey, items, 10);
  return items;
}

/**
 * Expand a location string into an array of normalized tokens to match against JLL data.
 * "OMR" â†’ ['omr', 'old mahabalipuram', 'sholinganallur', ...]
 * "T Nagar" â†’ ['t nagar', 't nagar', 'tnagar', 'teynampet']  (dot variants normalized)
 * "Velachery" â†’ ['velachery']  (no alias, returned as-is)
 *
 * All returned tokens are already normalized (no dots/hyphens) so they can be
 * compared directly against normalizeStr(haystack).
 */
function expandLocationTokens(location) {
  const key = normalizeStr(location);
  // Direct alias hit (try both raw and normalized key)
  if (LOCATION_ALIASES[key]) return LOCATION_ALIASES[key].map(normalizeStr);
  // Also try the original lower-trimmed key (alias keys may have dots)
  const rawKey = location.toLowerCase().trim();
  if (LOCATION_ALIASES[rawKey]) return LOCATION_ALIASES[rawKey].map(normalizeStr);
  // Partial alias hit (e.g. "OMR Road" â†’ key contains "omr")
  for (const [alias, expansions] of Object.entries(LOCATION_ALIASES)) {
    const normAlias = normalizeStr(alias);
    if (key.includes(normAlias) || normAlias.includes(key)) return expansions.map(normalizeStr);
  }
  // No alias â€” keep multi-word phrase as ONE token. Do NOT split on spaces.
  // Splitting "Gandhi Nagar" â†’ ["ghandi","nagar"] causes "nagar" to match
  // Anna Nagar, T. Nagar, and dozens of other unrelated locations.
  // Only split on explicit separators (comma, slash) for multi-area requests
  // like "OMR, ECR" â†’ ["omr", "ecr"].
  const parts = key.split(/[,\/]+/).map(s => s.trim()).filter(t => t.length >= 3);
  return parts.length > 0 ? parts : [key].filter(t => t.length >= 3);
}

// â”€â”€â”€ JLL Catalog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Loaded once at startup from backend/src/data/jll-catalog.json (generated by
// npm run sync-catalog). Contains all canonical Location, Micro_Market, City,
// and Project_Type values as stored in the JLL database.
//
// Purpose: map user-spoken names ("T Nagar", "apartments") to the exact JLL
// format ("T. Nagar", "Apartments") so the API returns correct results.
const CATALOG_PATH = path.join(__dirname, '../data/jll-catalog.json');
let _catalog = null;

function loadCatalog() {
  if (_catalog) return _catalog;
  try {
    const raw = fs.readFileSync(CATALOG_PATH, 'utf8');
    _catalog = JSON.parse(raw);
    const cityCount = (_catalog.cities || []).length;
    if (cityCount > 0) {
      console.log(`[Catalog] Loaded JLL catalog: ${cityCount} cities, synced_at=${_catalog.synced_at}`);
    } else {
      console.log('[Catalog] JLL catalog is empty â€” run: npm run sync-catalog');
    }
  } catch (err) {
    console.warn(`[Catalog] Could not load catalog (${err.message}) â€” fuzzy matching disabled`);
    _catalog = { cities: [], by_city: {}, property_types: [] };
  }
  return _catalog;
}

// Reload catalog from disk (called after sync-catalog updates the file)
function reloadCatalog() {
  _catalog = null;
  return loadCatalog();
}

/**
 * Resolve a user-spoken value to its canonical JLL stored name.
 *
 * Strategy (in order):
 *   1. Exact normalized match:    "t nagar" === normalizeStr("T. Nagar") â†’ "T. Nagar"
 *   2. Starts-with match:         "omr" matches "OMR" â†’ first hit
 *   3. Contains match:            "nagar" substring in candidate
 *   4. Candidate contains input:  "anna nagar" contains "nagar" â€” return most specific hit
 *   5. No match â†’ return null (caller uses the original input or alias expansion)
 *
 * @param {string} input  â€” user spoken value, e.g. "T Nagar", "apartments", "chennai"
 * @param {string} field  â€” "location" | "micro_market" | "city" | "property_type"
 * @param {string} [city] â€” for location/micro_market, restrict to this city's catalog
 * @returns {string|null} canonical JLL name, or null if no confident match
 */
function resolveCatalogName(input, field, city = null) {
  if (!input) return null;
  const catalog = loadCatalog();
  const norm = normalizeStr(input);

  let candidates = [];
  if (field === 'city') {
    candidates = catalog.cities || [];
  } else if (field === 'property_type') {
    // Merge city-specific and global types
    const global = catalog.property_types || [];
    const cityTypes = city && catalog.by_city?.[city]?.property_types || [];
    candidates = [...new Set([...global, ...cityTypes])];
  } else if (field === 'location') {
    // Try city-specific first, then all cities combined
    if (city && catalog.by_city?.[city]?.locations) {
      candidates = catalog.by_city[city].locations;
    } else {
      candidates = Object.values(catalog.by_city || {})
        .flatMap(c => c.locations || []);
    }
  } else if (field === 'micro_market') {
    if (city && catalog.by_city?.[city]?.micro_markets) {
      candidates = catalog.by_city[city].micro_markets;
    } else {
      candidates = Object.values(catalog.by_city || {})
        .flatMap(c => c.micro_markets || []);
    }
  }

  if (candidates.length === 0) return null;

  // 1. Exact normalized match
  const exact = candidates.find(c => normalizeStr(c) === norm);
  if (exact) return exact;

  // 2. Starts-with match (e.g. "anna nagar" â†’ "Anna Nagar West" if that's the only one)
  //    But prefer shorter/exact over longer
  const starts = candidates.filter(c => normalizeStr(c).startsWith(norm));
  if (starts.length === 1) return starts[0];
  // If multiple start-with hits, return the shortest (most specific name without suffix)
  if (starts.length > 1) {
    const shortest = starts.reduce((a, b) => a.length <= b.length ? a : b);
    return shortest;
  }

  // 3. Input contains candidate (user said more than the stored name â€” trim suffix)
  //    e.g. user says "T Nagar area" â†’ norm contains "t nagar"
  const inputContains = candidates.filter(c => norm.includes(normalizeStr(c)));
  if (inputContains.length > 0) {
    // Return the longest match (most specific)
    return inputContains.reduce((a, b) => a.length >= b.length ? a : b);
  }

  // 4. Candidate contains input (user said part of the stored name)
  //    e.g. user says "velachery" â†’ stored "Velachery" â†’ exact match above catches this
  //    but covers partial: user says "sholingan" â†’ stored "Sholinganallur"
  const candContains = candidates.filter(c => normalizeStr(c).includes(norm));
  if (candContains.length === 1) return candContains[0];
  // Multiple partial hits â€” return only if norm is long enough to be unambiguous
  if (candContains.length > 1 && norm.length >= 5) {
    return candContains.reduce((a, b) => a.length <= b.length ? a : b);
  }

  return null;
}

function getAreaMinPrice(cityData, areaName) {
  if (!cityData || !areaName) return null;
  const areaMap = cityData.area_min_prices || {};
  if (areaMap[areaName] != null) return areaMap[areaName];

  const norm = normalizeStr(areaName);
  for (const [key, value] of Object.entries(areaMap)) {
    if (normalizeStr(key) === norm) return value;
  }
  return null;
}

// â”€â”€â”€ GET|POST /api/integration/proxy/search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Accepts params via query string (GET) or JSON body (POST).
// Uses catalog to resolve user-spoken names to exact JLL format before API call.
//
// Pagination strategy:
//   - Voice agent sends page=1 on first call, page=2/3/... on "show more"
//   - We always fetch limit=20 from JLL per page (for location filtering headroom)
//   - Client-side location filter (with alias expansion) narrows results
//   - exclude_slugs (JSON array) removes already-shown properties (dedup across pages)
//   - We always return exactly PAGE_SIZE=3 results to the voice agent
//   - has_more=true when JLL returned a full page (likely more pages exist)
const PAGE_SIZE = 3;

async function handleSearch(req, res) {
  const t0 = Date.now();
  // Support both GET (query params) and POST (JSON body)
  const p = Object.assign({}, req.query, req.body);
  const {
    city, property_type, max_price, min_price,
    project_category, construction_state,
    location, page = '1',
    bedrooms,          // e.g., "2", "3" â€” filter by BHK count
    exclude_slugs,
    sort_preference,   // "price_asc" | "price_desc" | "default" â€” set by voice agent from buyer persona
    call_id, assistant_id, org_id,
  } = p;

  // â”€â”€ Cache lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Skip cache when exclude_slugs is set (pagination dedup varies per call) or
  // when this is a pure city-wide search with no location (rare, fast anyway).
  const _ck = _cacheKey(p);
  const _skipCache = !!(exclude_slugs && String(exclude_slugs).length > 2);
  if (!_skipCache) {
    const _cached = _getCached(_ck);
    if (_cached) {
      console.log(`[Integration] cache HIT for key=${_ck.slice(0, 60)}... (${Date.now() - t0}ms)`);
      _logBackendProperties('cache hit', _cached?.data);
      return res.json(_cached);
    }
  }

  const pageNum = Math.max(1, parseInt(page) || 1);

  // Parse exclude_slugs â€” JSON array string or comma-separated
  let excludeSet = new Set();
  if (exclude_slugs) {
    try {
      const parsed = JSON.parse(exclude_slugs);
      if (Array.isArray(parsed)) parsed.forEach(s => excludeSet.add(String(s)));
    } catch {
      String(exclude_slugs).split(',').forEach(s => { if (s.trim()) excludeSet.add(s.trim()); });
    }
  }

  // â”€â”€ Catalog resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Map user-spoken values to exact JLL stored names before API calls.
  // e.g. "T Nagar" â†’ "T. Nagar", "apartments" â†’ "Apartments", "bengalore" â†’ "Bengaluru"
  // If catalog is empty (not yet synced), values pass through unchanged.

  // 1. City â€” resolve first since location/micro_market resolution is city-scoped
  const resolvedCity = resolveCatalogName(city, 'city') || city || '';
  if (resolvedCity !== city) {
    console.log(`[Catalog] city: "${city}" â†’ "${resolvedCity}"`);
  }

  // 2. Location â€” resolve against catalog for the resolved city
  let resolvedLocation = location || '';
  let resolvedMicroMarket = '';
  if (location) {
    const catalogLoc = resolveCatalogName(location, 'location', resolvedCity);
    if (catalogLoc) {
      resolvedLocation = catalogLoc;
      if (catalogLoc !== location) {
        console.log(`[Catalog] location: "${location}" â†’ "${resolvedLocation}"`);
      }
    }
    const catalogMicroMarket = resolveCatalogName(location, 'micro_market', resolvedCity);
    if (catalogMicroMarket) {
      resolvedMicroMarket = catalogMicroMarket;
      if (catalogMicroMarket !== location) {
        console.log(`[Catalog] micro_market: "${location}" -> "${resolvedMicroMarket}"`);
      }
    }
    // If no catalog match, resolvedLocation stays as the raw user input
    // (alias expansion + client-side filter handles corridor names like OMR)
  }

  // 3. Property type
  let resolvedType = property_type || '';
  if (property_type) {
    const catalogType = resolveCatalogName(property_type, 'property_type', resolvedCity);
    if (catalogType) {
      resolvedType = catalogType;
      if (catalogType !== property_type) {
        console.log(`[Catalog] property_type: "${property_type}" â†’ "${resolvedType}"`);
      }
    }
  }

  console.log(
    `[Integration] search_properties â†’ city=${resolvedCity} page=${pageNum} ` +
    `location=${resolvedLocation || '-'} type=${resolvedType || '-'} ` +
    `category=${project_category || '-'} ` +
    `construction=${construction_state || '-'} ` +
    `bedrooms=${bedrooms || '-'} ` +
    `budget=${min_price || 0}-${max_price || 'âˆž'} ` +
    `exclude=${excludeSet.size} call=${call_id || '-'}`
  );

  const _catalogDebug = loadCatalog();
  const _debugCityData = _catalogDebug.by_city?.[resolvedCity] || _catalogDebug.by_city?.[city] || {};
  const _tNagarResolved = resolveCatalogName('T Nagar', 'location', resolvedCity) || resolveCatalogName('T Nagar', 'micro_market', resolvedCity) || 'none';
  console.log(
    `[Integration] locality debug: area_min_prices_loaded=${!!_debugCityData.area_min_prices} ` +
    `city=${resolvedCity || city || '-'} input_location=${location || '-'} ` +
    `resolved_location=${resolvedLocation || '-'} t_nagar_resolves_to=${_tNagarResolved}`
  );

  const isCatalogLocality = !!(resolvedLocation || resolvedMicroMarket) && !!loadCatalog().by_city?.[resolvedCity || city];
  const jllLocationParam = isCatalogLocality ? resolvedLocation : (location || '');
  let usedClientSideFilter = false;
  const _catalogForBudget = loadCatalog();
  const _cityDataForBudget = _catalogForBudget.by_city?.[resolvedCity] || _catalogForBudget.by_city?.[city] || {};
  const _catalogAreaMin = getAreaMinPrice(_cityDataForBudget, resolvedLocation);

  if (_catalogAreaMin != null && max_price && Number(max_price) < _catalogAreaMin) {
    // â”€â”€ Verify property type exists in this area before declaring budget_too_low â”€
    // The catalog's area_min_prices is across ALL property types, so we need to check
    // if properties of the requested type actually exist in this location.
    let propertyTypeExistsInArea = true; // default to true if no property type specified
    if (resolvedType && resolvedLocation) {
      try {
        const verifyQs = new URLSearchParams({
          city: resolvedCity || '',
          location: resolvedLocation,
          page: '1',
          limit: '1',
        });
        verifyQs.set('property_type', resolvedType);
        // No budget filters - just check if ANY properties of this type exist here
        const verifyRes = await fetch(`${JLL_BASE}/api/user/search/projects?${verifyQs}`, { timeout: 5000 });
        const verifyJson = await verifyRes.json();
        const verifyItems = Array.isArray(verifyJson?.data) ? verifyJson.data : [];
        propertyTypeExistsInArea = verifyItems.length > 0;
        console.log(`[Integration] budget_too_low check: ${resolvedType} in '${resolvedLocation}' â†’ exists=${propertyTypeExistsInArea}, count=${verifyItems.length}`);
      } catch (verifyErr) {
        // If verification fails, proceed with caution (assume exists)
        console.warn(`[Integration] Property type verification failed: ${verifyErr.message}`);
        propertyTypeExistsInArea = true;
      }
    }

    // If property type doesn't exist in this area, DON'T return budget_too_low
    // Let the normal search flow handle it (will return zero results / no_location_match)
    if (!propertyTypeExistsInArea) {
      console.log(`[Integration] ${resolvedType} does not exist in '${resolvedLocation}' â€” skipping budget_too_low short-circuit`);
      // Continue to normal search flow below
    } else {
      const areaMinPrice = formatPriceINR(_catalogAreaMin);
      
      // â”€â”€ Find budget-matching areas (not just nearby areas) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Search city-wide to find areas that actually have properties within user's budget
    const budgetMatchingAreas = [];
    const MAX_AREA_SCAN_PAGES = 10;
    const MIN_RESULTS_THRESHOLD = 3;

    try {
      // Phase 1: fetch page 1 first to check has_more and initial results
      const firstQs = new URLSearchParams({
        city: resolvedCity || '',
        page: '1',
        limit: '20',
      });
      if (resolvedType) firstQs.set('property_type', resolvedType);
      if (max_price) firstQs.set('max_price', String(max_price));
      if (min_price) firstQs.set('min_price', String(min_price));

      const firstRes = await fetch(`${JLL_BASE}/api/user/search/projects?${firstQs}`, { timeout: 8000 });
      const firstJson = await firstRes.json();
      const firstItems = Array.isArray(firstJson?.data) ? firstJson.data : [];

      // Process page 1 results
      for (const item of firstItems) {
        const itemLocation = item.Location || item.Micro_Market || '';
        if (!itemLocation) continue;

        const configs = item.configurations || item.configs || [];
        const hasMatchingConfig = configs.some(c => {
          const price = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
          if (price <= 0) return false;
          if (min_price && price < Number(min_price)) return false;
          if (max_price && price > Number(max_price)) return false;
          return true;
        });

        if (hasMatchingConfig) {
          const normalizedArea = normalizeStr(itemLocation);
          if (!budgetMatchingAreas.find(a => normalizeStr(a) === normalizedArea)) {
            budgetMatchingAreas.push(itemLocation);
          }
        }
      }

      // Phase 2: fire remaining pages in parallel only if needed
      const hasMore = firstItems.length >= 20;
      if (hasMore && budgetMatchingAreas.length < MIN_RESULTS_THRESHOLD) {
        const remainingPages = [];
        for (let pg = 2; pg <= MAX_AREA_SCAN_PAGES; pg++) remainingPages.push(pg);

        const pageResults = await Promise.all(
          remainingPages.map(pg => {
            const qs = new URLSearchParams({
              city: resolvedCity || '',
              page: String(pg),
              limit: '20',
            });
            if (resolvedType) qs.set('property_type', resolvedType);
            if (max_price) qs.set('max_price', String(max_price));
            if (min_price) qs.set('min_price', String(min_price));

            return fetch(`${JLL_BASE}/api/user/search/projects?${qs}`, { timeout: 8000 })
              .then(r => r.json())
              .catch(() => ({ data: [] }));
          })
        );

        // Process parallel results
        for (const pr of pageResults) {
          const items = Array.isArray(pr?.data) ? pr.data : [];
          if (items.length === 0) continue;

          for (const item of items) {
            const itemLocation = item.Location || item.Micro_Market || '';
            if (!itemLocation) continue;

            const configs = item.configurations || item.configs || [];
            const hasMatchingConfig = configs.some(c => {
              const price = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
              if (price <= 0) return false;
              if (min_price && price < Number(min_price)) return false;
              if (max_price && price > Number(max_price)) return false;
              return true;
            });

            if (hasMatchingConfig) {
              const normalizedArea = normalizeStr(itemLocation);
              if (!budgetMatchingAreas.find(a => normalizeStr(a) === normalizedArea)) {
                budgetMatchingAreas.push(itemLocation);
              }
            }
          }
        }
      }
    } catch (err) {
      console.warn(`[Integration] budget-matching area scan failed: ${err.message}`);
    }
    
    // Filter out current location and limit results
    const _suggestedAreas = budgetMatchingAreas
      .filter(a => normalizeStr(a) !== normalizeStr(resolvedLocation))
      .slice(0, 5);

    console.log(`[Integration] budget_too_low short-circuit for '${resolvedLocation}' at max_price=${max_price}, area_min_price=${areaMinPrice}, found ${budgetMatchingAreas.length} budget-matching areas`);
    console.log(`[Integration] budget_too_low â† returning data=[] suggested_areas=[${_suggestedAreas.join(', ')}]`);

    return res.json({
      success: true,
      data: [],
      has_more: false,
      current_page: pageNum,
      budget_too_low: true,
      area_min_price: areaMinPrice,
      searched_location: location || resolvedLocation,
      suggested_areas: _suggestedAreas,
      message: `Properties in '${location || resolvedLocation}' start from ${areaMinPrice}.`,
    });
    } // closes else block (propertyTypeExistsInArea)
  } // closes if (_catalogAreaMin != null...) block

  // Helper: fetch one page from JLL and return { items, total }
  async function fetchJllPage(pg, locationOverride = undefined) {
    const qs = new URLSearchParams({ city: resolvedCity || '', page: String(pg), limit: '20' });
    if (resolvedType)     qs.set('property_type', resolvedType);
    if (max_price)        qs.set('max_price', String(max_price));
    if (min_price)        qs.set('min_price', String(min_price));
    if (locationOverride) qs.set('location', locationOverride);
    const jllRes = await fetch(`${JLL_BASE}/api/user/search/projects?${qs}`, { timeout: 10000 });
    const json   = await jllRes.json();
    const items  = Array.isArray(json?.data) ? json.data : (Array.isArray(json) ? json : []);
    const total  = json?.total ?? json?.totalCount ?? json?.total_count ??
                   json?.count ?? json?.totalRecords ?? null;
    // Log top-level keys once (omit 'data' to keep logs clean)
    if (json && typeof json === 'object' && !Array.isArray(json)) {
      const keys = Object.keys(json).filter(k => k !== 'data');
      if (keys.length) console.log(`[Integration] JLL top-level keys (pg ${pg} loc=${locationOverride || false}): ${keys.join(', ')} | total=${total}`);
    }
    return { items, total };
  }

  let raw = [], error = null, jllTotal = null;
  try {
    const directQueryVariants = [
      location || '',
      resolvedLocation || '',
      resolvedMicroMarket || '',
    ].filter(Boolean).filter((value, index, arr) =>
      arr.findIndex(v => String(v).trim().toLowerCase() === String(value).trim().toLowerCase()) === index
    );

    if (resolvedLocation || resolvedMicroMarket) {
      for (const variant of directQueryVariants) {
        console.log(`[Integration] direct JLL locality try: '${variant}'`);
        const attempt = await fetchJllPage(pageNum, variant);
        if (attempt.items.length > 0) {
          raw = attempt.items;
          jllTotal = attempt.total;
          break;
        }
      }
    } else {
      const first = await fetchJllPage(pageNum, jllLocationParam || undefined);
      raw      = first.items;
      jllTotal = first.total;
    }

    // Only non-catalog/freeform locations should fall back to city-wide pagination.
    if (!isCatalogLocality && resolvedLocation && raw.length === 0) {
      console.log(`[Integration] JLL returned 0 with location='${jllLocationParam}' â€” retrying without location param`);
      const fallback = await fetchJllPage(pageNum, undefined);
      raw      = fallback.items;
      jllTotal = fallback.total;
      usedClientSideFilter = true;
    }
  } catch (err) {
    error = err.message;
    console.error(`[Integration] JLL fetch error: ${error}`);
  }

  // â”€â”€ Step 1: Location filter with alias expansion + dot-normalization â”€â”€â”€â”€â”€â”€
  // JLL field values use dotted abbreviations: "T. Nagar", "Hi-Tech City".
  // Spoken input is plain: "T Nagar", "Hitech City".
  // normalizeStr() strips dots/hyphens from BOTH sides before comparing, so
  // "t. nagar" and "t nagar" both normalize to "t nagar" â†’ match succeeds.
  //
  // Accumulation strategy (latency-optimised):
  //   1. Fast path: if page 1 already has â‰¥ PAGE_SIZE matches â†’ done immediately.
  //   2. Parallel burst: when page 1 has < PAGE_SIZE matches, fetch the next
  //      MAX_ACCUMULATE_PAGES pages IN PARALLEL using Promise.all instead of
  //      sequential awaits. This reduces worst-case from 15 Ã— 200ms = 3s to
  //      just 1 + 1 batch = ~400ms.
  //   3. Hard cap: 2 batches Ã— 4 pages = 8 extra pages maximum.
  //      If still < PAGE_SIZE results, the area is genuinely sparse in JLL's dataset.

  let filtered = raw;
  const matchesProjectCategory = (p) => {
    if (!project_category) return true;
    const wanted = normalizeStr(String(project_category || ''));
    const actual = normalizeStr(String(p.Project_Category || ''));
    if (!wanted || !actual) return false;
    if (wanted === 'new') return actual.includes('new');
    if (wanted === 'resale') return actual.includes('resale');
    return actual === wanted;
  };
  const matchesConstructionState = (p) => {
    if (!construction_state) return true;
    const wanted = normalizeStr(String(construction_state || ''));
    const actual = normalizeStr(String(p.State_Of_Construction || ''));
    if (!wanted || !actual) return false;
    if (wanted === 'ready_to_move') {
      return actual.includes('ready') || actual.includes('move');
    }
    if (wanted === 'under_construction') {
      return actual.includes('under') || actual.includes('construction');
    }
    if (wanted === 'launched') {
      return actual.includes('launch');
    }
    return actual === wanted;
  };
  const matchesBedrooms = (p) => {
    if (!bedrooms) return true;
    const targetBhk = parseInt(bedrooms, 10);
    if (isNaN(targetBhk)) return true;

    const configs = p.configurations || p.configs || [];
    return configs.some(c => {
      const configType = c.Config_Type || '';
      const match = configType.match(/(\d+(?:\.\d+)?)\s*BHK/i);
      if (!match) return false;
      const configBhk = parseFloat(match[1]);
      return Math.floor(configBhk) === targetBhk;
    });
  };
  if (isCatalogLocality && resolvedLocation && raw.length > 0) {
    const locTokens = [
      ...new Set([
        ...expandLocationTokens(resolvedLocation),
        ...(location && location !== resolvedLocation ? expandLocationTokens(location) : []),
      ])
    ];
    const exactFiltered = raw.filter(p => {
      // Only match against canonical location fields â€” not project name or address
      // (prevents "near T. Nagar" in a project name causing false positives)
      const locationFields = [p.Location || '', p.Area || '', p.Micro_Market || ''];
      const locHay = normalizeStr(locationFields.join(' '));
      const locationMatch = locTokens.some(t => {
        const escaped = t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const startBoundaryRegex = new RegExp(`(^|\\s)${escaped}(\\s|_|\\b)`, 'i');
        const fullWordRegex = new RegExp(`(^|\\s)${escaped}(\\s|$)`, 'i');
        if (t.includes(' ')) return startBoundaryRegex.test(locHay);
        return fullWordRegex.test(locHay);
      });
      return locationMatch && matchesBedrooms(p) && matchesProjectCategory(p) && matchesConstructionState(p);
    });
    filtered = exactFiltered;
    console.log(`[Integration] exact locality filter for '${resolvedLocation}': ${filtered.length}/${raw.length} matched`);
    if (!filtered.length) {
      const dropped = raw.slice(0, 5).map(p =>
        `${p.Project_Name || '?'} | loc=${p.Location || '-'} | area=${p.Area || '-'} | mm=${p.Micro_Market || '-'}`
      );
      console.log(
        `[Integration] exact locality dropped rows for '${resolvedLocation}' ` +
        `tokens=[${locTokens.join(', ')}]:\n  ${dropped.join('\n  ')}`
      );
    }
  } else if (resolvedLocation && raw.length > 0) {
    // Build tokens from the resolved location (may differ from user's raw input)
    // Also include the original user input as fallback tokens
    const locTokens = [
      ...new Set([
        ...expandLocationTokens(resolvedLocation),
        ...(location && location !== resolvedLocation ? expandLocationTokens(location) : []),
      ])
    ];
    console.log(`[Integration] location tokens for '${resolvedLocation}': [${locTokens.join(', ')}]`);

    // Log actual Location/Micro_Market values from first batch for diagnostics
    const sampleFields = raw.slice(0, 5).map(p =>
      `${p.Project_Name || '?'} | loc=${p.Location || '-'} | mm=${p.Micro_Market || '-'}`
    );
    console.log(`[Integration] JLL sample fields (page ${pageNum}):\n  ${sampleFields.join('\n  ')}`);

    // Dot-normalized filter: strips punctuation from field values before matching
    // Smart matching: 'anna nagar' matches 'anna nagar west' but 'nagar' won't match 'thirumangalam'
    // IMPORTANT: Only match against Location/Area/Micro_Market fields â€” NOT Project_Name or
    // Address_Line_1. Including those causes false positives like a Porur project named
    // "near OMR corridor" incorrectly appearing in OMR search results.

    const filterBatch = (batch) => batch.filter(p => {
      const hay = normalizeStr([
        p.Location || '', p.Area || '', p.Micro_Market || '',
      ].join(' '));
      const locationMatch = locTokens.some(t => {
        // Escape regex special chars in token
        const escaped = t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        // Match if: (1) token is at word boundary at START, or (2) complete word match
        // This allows 'anna nagar' to match 'anna nagar west' but blocks 'nagar' matching 'thirumangalam'
        const startBoundaryRegex = new RegExp(`(^|\\s)${escaped}(\\s|_|\\b)`, 'i');
        const fullWordRegex = new RegExp(`(^|\\s)${escaped}(\\s|$)`, 'i');
        // For multi-word tokens (like "anna nagar"), allow start-boundary match
        // For single-word tokens, require full word match unless it's the start of a longer phrase
        if (t.includes(' ')) {
          return startBoundaryRegex.test(hay);
        }
        // Single word: must be full word match (prevents 'nagar' in 'thirumangalam')
        return fullWordRegex.test(hay);
      });
      // Also filter by bedroom count if specified
      return locationMatch && matchesBedrooms(p) && matchesProjectCategory(p) && matchesConstructionState(p);
    });

    let matched = filterBatch(raw);
    console.log(`[Integration] location filter (page ${pageNum}): ${matched.length}/${raw.length} matched`);

    // Accumulate more pages when we have fewer than PAGE_SIZE matches.
    // This handles two cases:
    //   (a) 0 matches â€” area token not found in this batch at all
    //   (b) Few matches â€” e.g. Electronic City matches only 1/20 per page
    //       because results are scattered across the full city dataset.
    //
    // Two-batch parallel strategy (latency-optimised with safety fallback):
    //   Batch 1: pages [pageNum+1 â€¦ pageNum+BATCH_SIZE] fetched in parallel.
    //   If still < PAGE_SIZE after batch 1, fire Batch 2 (next BATCH_SIZE pages).
    //   Hard cap: 2 batches = MAX_ACCUMULATE_PAGES pages total.
    //   Worst-case latency: page1 (serial) + batch1 (parallel) + batch2 (parallel)
    //   = ~3 round-trips â‰ˆ 600ms, vs old 15 sequential â‰ˆ 3,000ms.
    //   If the area genuinely doesn't exist in JLL (all pages exhausted = 0 matches),
    //   the budget_too_low / no_location_match logic downstream handles it correctly.
    const BATCH_SIZE = 4;  // pages per parallel burst
    if (matched.length < PAGE_SIZE && !error) {
      usedClientSideFilter = true;
      const seenSlugs = new Set([
        ...raw.map(p => p.Project_Slug).filter(Boolean),
        ...excludeSet,
      ]);

      console.log(
        matched.length === 0
          ? `[Integration] 0 matches â€” parallel accumulation for '${resolvedLocation}'`
          : `[Integration] ${matched.length} matches â€” need more, parallel accumulation for '${resolvedLocation}'`
      );

      /**
       * Fetch a batch of pages in parallel and accumulate matches.
       * Returns true if more pages might still have results (none of the batch was empty).
       */
      async function runBatch(startPage) {
        const pageNums = [];
        for (let ep = startPage; ep < startPage + BATCH_SIZE; ep++) pageNums.push(ep);

        const results = await Promise.allSettled(pageNums.map(ep => fetchJllPage(ep, undefined)));
        let anyNonEmpty = false;

        for (let i = 0; i < results.length; i++) {
          const ep = pageNums[i];
          const r = results[i];
          if (r.status === 'rejected') {
            console.error(`[Integration] page ${ep} failed: ${r.reason?.message}`);
            continue;
          }
          const next = r.value;
          if (next.items.length === 0) {
            console.log(`[Integration] page ${ep}: empty â€” JLL dataset exhausted`);
            continue; // don't set anyNonEmpty â€” JLL has no more data
          }
          anyNonEmpty = true;
          if (next.total != null) jllTotal = next.total;
          const newItems = next.items.filter(p => !seenSlugs.has(p.Project_Slug));
          next.items.forEach(p => { if (p.Project_Slug) seenSlugs.add(p.Project_Slug); });
          const newMatches = filterBatch(newItems);
          matched = [...matched, ...newMatches];
          console.log(`[Integration] page ${ep} (parallel): +${newMatches.length} â†’ total ${matched.length}`);
        }
        return anyNonEmpty;
      }

      // Batch 1
      const batch1Start = pageNum + 1;
      const moreDataAfterBatch1 = await runBatch(batch1Start);

      // Batch 2 â€” only if batch 1 found non-empty pages AND we still need more results
      if (matched.length < PAGE_SIZE && moreDataAfterBatch1) {
        console.log(`[Integration] still ${matched.length}/${PAGE_SIZE} â€” firing batch 2`);
        await runBatch(batch1Start + BATCH_SIZE);
      }
    }

    if (matched.length > 0) {
      filtered = matched;
      console.log(`[Integration] location filter final: ${matched.length} matches accumulated`);
    } else {
      // Exhausted all pages â€” no match found anywhere.
      // Before returning no_location_match, distinguish two cases:
      //   A) Budget too low: area EXISTS in catalog but user's budget can't afford it
      //      (T. Nagar, Anna Nagar etc. appear on pages 10â€“20+ when sorted cheapest-first,
      //       so a single-page scan always misses them â€” use catalog check instead)
      //   B) Area doesn't exist: location is not in catalog â†’ genuine no_location_match
      let budgetTooLow = false;
      let areaMinPrice  = null;
      const budgetWasActive = !!(max_price || min_price);

      if (budgetWasActive) {
        // â”€â”€ Step 1: Catalog-first existence check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        // Check if the location actually exists in our catalog before hitting
        // the JLL API again. Expensive areas like T. Nagar appear on page 15+
        // when sorted by price ascending â€” a single-page scan always misses them.
        // The catalog check is instant and authoritative.
        const _catalog   = loadCatalog();
        const _cityData  = _catalog.by_city?.[resolvedCity] || _catalog.by_city?.[city] || {};
        const _allCatalogLocs = [
          ...(_cityData.locations    || []),
          ...(_cityData.micro_markets || []),
        ];
        const _normResLoc        = normalizeStr(resolvedLocation);
        const _locTokensExpanded = expandLocationTokens(resolvedLocation);

        // A location is "in catalog" if any catalog entry matches via
        // normalized equality OR the alias expansion overlaps with catalog entries
        const locationExistsInCatalog = _allCatalogLocs.some(loc => {
          const n = normalizeStr(loc);
          return (
            n === _normResLoc ||
            n.includes(_normResLoc) ||
            _normResLoc.includes(n) ||
            _locTokensExpanded.some(t => n.includes(t) || t.includes(n))
          );
        });

        console.log(
          `[Integration] catalog existence check for '${resolvedLocation}' ` +
          `(normalized='${_normResLoc}'): ${locationExistsInCatalog ? 'EXISTS' : 'NOT FOUND'} ` +
          `(checked ${_allCatalogLocs.length} catalog entries for ${resolvedCity})`
        );

        if (locationExistsInCatalog) {
          // Location exists in catalog â†’ budget is the problem, not a wrong area name.
          // Mark budget_too_low immediately (no API needed for this decision).
          budgetTooLow = true;

          const catalogAreaMin = getAreaMinPrice(_cityData, resolvedLocation);
          if (catalogAreaMin) {
            areaMinPrice = formatPriceINR(catalogAreaMin);
            console.log(`[Integration] area_min_price from catalog for '${resolvedLocation}' = ${areaMinPrice}`);
          }

          // â”€â”€ Step 2: Scan up to 10 pages city-wide (no budget filter) â”€â”€â”€â”€â”€â”€â”€
          // Find the actual cheapest price in this area so the agent can say
          // "Properties in T. Nagar start from X Cr" rather than a vague message.
          const MAX_PRICE_SCAN_PAGES = 25;
          console.log(`[Integration] budget_too_low=true for '${resolvedLocation}' â€” scanning up to ${MAX_PRICE_SCAN_PAGES} pages for area min price`);

          for (let pg = 1; pg <= MAX_PRICE_SCAN_PAGES && !areaMinPrice; pg++) {
            try {
              const scanQs = new URLSearchParams({
                city:  resolvedCity || '',
                page:  String(pg),
                limit: '20',
              });
              if (resolvedType) scanQs.set('property_type', resolvedType);
              // No price filter â€” we want all price points to find the minimum

              const scanRes  = await fetch(`${JLL_BASE}/api/user/search/projects?${scanQs}`, { timeout: 8000 });
              const scanJson = await scanRes.json();
              const scanItems = Array.isArray(scanJson?.data) ? scanJson.data : [];

              if (scanItems.length === 0) {
                console.log(`[Integration] price-scan: page ${pg} empty â€” stopping`);
                break;
              }

              const scanMatches = filterBatch(scanItems);
              console.log(`[Integration] price-scan page ${pg}: ${scanMatches.length}/${scanItems.length} matched '${resolvedLocation}'`);

              if (scanMatches.length > 0) {
                const allPrices = scanMatches.flatMap(p =>
                  (p.configurations || p.configs || [])
                    .map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0))
                    .filter(n => n > 0)
                );
                if (allPrices.length > 0) {
                  areaMinPrice = formatPriceINR(Math.min(...allPrices));
                  console.log(`[Integration] price-scan: found area min price = ${areaMinPrice} (page ${pg})`);
                  break; // Found the min â€” no need to scan more pages
                }
              }
            } catch (scanErr) {
              console.warn(`[Integration] price-scan page ${pg} failed: ${scanErr.message}`);
              break;
            }
          }

          // â”€â”€ Step 3: Catalog city-level fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
          // If we couldn't find area price in 10 pages, use catalog city min as proxy
          if (!areaMinPrice) {
            console.log(`[Integration] price-scan: no verified area minimum found for '${resolvedLocation}'`);
          }

          console.log(`[Integration] budget_too_low=true for '${resolvedLocation}', area_min_price=${areaMinPrice}`);

          // â”€â”€ Step 3: AUTOMATIC FALLBACK SEARCH without budget filter â”€â”€â”€â”€â”€â”€â”€
          // When user's budget doesn't match any properties, auto-search without budget
          // and return available properties with their actual prices
          console.log(`[Integration] Auto-fallback search: fetching properties in '${resolvedLocation}' without budget filter`);

          const fallbackResults = [];
          let areaMaxPrice = null;
          const MAX_FALLBACK_PAGES = 10;

          try {
            // Phase 1: fetch page 1 first
            const firstQs = new URLSearchParams({
              city:  resolvedCity || '',
              page:  '1',
              limit: '20',
            });
            if (resolvedType) firstQs.set('property_type', resolvedType);

            const firstRes = await fetch(`${JLL_BASE}/api/user/search/projects?${firstQs}`, { timeout: 8000 });
            const firstJson = await firstRes.json();
            const firstItems = Array.isArray(firstJson?.data) ? firstJson.data : [];

            if (firstItems.length > 0) {
              const firstMatches = filterBatch(firstItems);
              console.log(`[Integration] fallback-search page 1: ${firstMatches.length}/${firstItems.length} matched '${resolvedLocation}'`);

              for (const p of firstMatches) {
                const prices = (p.configurations || p.configs || [])
                  .map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0))
                  .filter(n => n > 0);
                if (prices.length > 0) {
                  const pMin = Math.min(...prices);
                  const pMax = Math.max(...prices);
                  if (!areaMaxPrice || pMax > areaMaxPrice) areaMaxPrice = pMax;
                }
              }
              fallbackResults.push(...firstMatches);
            }

            // Phase 2: fire remaining pages in parallel if needed
            const hasMore = firstItems.length >= 20 && fallbackResults.length < PAGE_SIZE;
            if (hasMore) {
              const remainingPages = [];
              for (let pg = 2; pg <= MAX_FALLBACK_PAGES; pg++) remainingPages.push(pg);

              const pageResults = await Promise.all(
                remainingPages.map(pg => {
                  const fbQs = new URLSearchParams({
                    city:  resolvedCity || '',
                    page:  String(pg),
                    limit: '20',
                  });
                  if (resolvedType) fbQs.set('property_type', resolvedType);

                  return fetch(`${JLL_BASE}/api/user/search/projects?${fbQs}`, { timeout: 8000 })
                    .then(r => r.json())
                    .catch(() => ({ data: [] }));
                })
              );

              // Process parallel results
              for (const pr of pageResults) {
                if (fallbackResults.length >= PAGE_SIZE) break;

                const fbItems = Array.isArray(pr?.data) ? pr.data : [];
                if (fbItems.length === 0) continue;

                const fbMatches = filterBatch(fbItems);
                if (fbMatches.length > 0) {
                  for (const p of fbMatches) {
                    const prices = (p.configurations || p.configs || [])
                      .map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0))
                      .filter(n => n > 0);
                    if (prices.length > 0) {
                      const pMin = Math.min(...prices);
                      const pMax = Math.max(...prices);
                      if (!areaMaxPrice || pMax > areaMaxPrice) areaMaxPrice = pMax;
                    }
                  }
                  fallbackResults.push(...fbMatches);
                }
              }
            }
          } catch (fbErr) {
            console.warn(`[Integration] fallback-search failed: ${fbErr.message}`);
          }

          console.log(`[Integration] fallback-search: found ${fallbackResults.length} properties in '${resolvedLocation}' (price range: ${areaMinPrice || 'unknown'} - ${areaMaxPrice ? formatPriceINR(areaMaxPrice) : 'unknown'})`);

          // If we found properties in the fallback search, return them with budget_mismatch_results flag
          if (fallbackResults.length > 0) {
            // Trim to PAGE_SIZE and normalize
            const trimmedFallback = fallbackResults.slice(0, PAGE_SIZE);
            const normalizedFallback = trimmedFallback.map(normalize);
            const _fallbackNames = normalizedFallback.map(p => p.name || p.project_name || '?').join(', ');
            console.log(`[Integration] budget_mismatch_results â† returning ${normalizedFallback.length} properties for '${resolvedLocation}': [${_fallbackNames}]`);

            const durationMs = Date.now() - t0;
            pool.query(
              `INSERT INTO integration_requests (call_id,assistant_id,org_id,tool_name,params,bedrooms,result_count,error,duration_ms)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
              [call_id||null, assistant_id||null, org_id||null, 'search_properties',
               JSON.stringify({ city: resolvedCity, location: resolvedLocation, property_type: resolvedType, max_price, min_price, page: pageNum, fallback: true }),
               bedrooms || null, normalizedFallback.length, 'budget_mismatch_results', durationMs]
            ).catch(e => console.error(`[Integration] DB log FAILED (${e.code || 'ERR'}): ${e.message}`));

            return res.json({
              success:          true,
              data:             normalizedFallback,
              has_more:         fallbackResults.length > PAGE_SIZE,
              current_page:     pageNum,
              budget_mismatch_results: true,  // Properties exist but outside user's budget
              user_min_price:   min_price || null,
              user_max_price:   max_price || null,
              area_min_price:   areaMinPrice,
              area_max_price:   areaMaxPrice ? formatPriceINR(areaMaxPrice) : null,
              searched_location: location || resolvedLocation,
              suggested_areas:  _suggestedAreas,
              search_scope:           'exact_locality',
              exact_location_searched: resolvedLocation || location || null,
              fallback_occurred:      false,  // results ARE from requested location, just over budget
              message: `Properties in '${location || resolvedLocation}' range from ${areaMinPrice || 'unknown'} to ${areaMaxPrice ? formatPriceINR(areaMaxPrice) : 'unknown'}, which is outside your budget of ${formatPriceINR(Number(min_price || 0)) || 'unspecified'} - ${formatPriceINR(Number(max_price || 0)) || 'unspecified'}.`,
            });
          }

          // If fallback also returned 0 results, proceed with original budget_too_low response
          console.log(`[Integration] fallback-search returned 0 results â€” returning budget_too_low`);
        } else {
          // Location not in catalog at all â†’ genuine "area doesn't exist" case
          console.log(`[Integration] '${resolvedLocation}' not in catalog for ${resolvedCity} â€” returning no_location_match`);
        }
      }

      const durationMs = Date.now() - t0;
      
      // â”€â”€ Find budget-matching areas for suggestions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      // When budget_too_low, suggest areas that actually have properties in user's budget
      let _suggestedAreas = [];
      if (budgetTooLow && (max_price || min_price)) {
        const budgetMatchingAreas = [];
        const MAX_SUGGEST_SCAN_PAGES = 8;

        try {
          // Phase 1: fetch page 1 first
          const firstQs = new URLSearchParams({
            city: resolvedCity || '',
            page: '1',
            limit: '20',
          });
          if (resolvedType) firstQs.set('property_type', resolvedType);
          if (max_price) firstQs.set('max_price', String(max_price));
          if (min_price) firstQs.set('min_price', String(min_price));

          const firstRes = await fetch(`${JLL_BASE}/api/user/search/projects?${firstQs}`, { timeout: 8000 });
          const firstJson = await firstRes.json();
          const firstItems = Array.isArray(firstJson?.data) ? firstJson.data : [];

          // Process page 1 results
          for (const item of firstItems) {
            const itemLocation = item.Location || item.Micro_Market || '';
            if (!itemLocation) continue;

            const configs = item.configurations || item.configs || [];
            const hasMatchingConfig = configs.some(c => {
              const price = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
              if (price <= 0) return false;
              if (min_price && price < Number(min_price)) return false;
              if (max_price && price > Number(max_price)) return false;
              return true;
            });

            if (hasMatchingConfig) {
              const normalizedArea = normalizeStr(itemLocation);
              if (!budgetMatchingAreas.find(a => normalizeStr(a) === normalizedArea)) {
                budgetMatchingAreas.push(itemLocation);
              }
            }
          }

          // Phase 2: fire remaining pages in parallel if needed
          const hasMore = firstItems.length >= 20;
          if (hasMore) {
            const remainingPages = [];
            for (let pg = 2; pg <= MAX_SUGGEST_SCAN_PAGES; pg++) remainingPages.push(pg);

            const pageResults = await Promise.all(
              remainingPages.map(pg => {
                const qs = new URLSearchParams({
                  city: resolvedCity || '',
                  page: String(pg),
                  limit: '20',
                });
                if (resolvedType) qs.set('property_type', resolvedType);
                if (max_price) qs.set('max_price', String(max_price));
                if (min_price) qs.set('min_price', String(min_price));

                return fetch(`${JLL_BASE}/api/user/search/projects?${qs}`, { timeout: 8000 })
                  .then(r => r.json())
                  .catch(() => ({ data: [] }));
              })
            );

            // Process parallel results
            for (const pr of pageResults) {
              const items = Array.isArray(pr?.data) ? pr.data : [];
              if (items.length === 0) continue;

              for (const item of items) {
                const itemLocation = item.Location || item.Micro_Market || '';
                if (!itemLocation) continue;

                const configs = item.configurations || item.configs || [];
                const hasMatchingConfig = configs.some(c => {
                  const price = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
                  if (price <= 0) return false;
                  if (min_price && price < Number(min_price)) return false;
                  if (max_price && price > Number(max_price)) return false;
                  return true;
                });

                if (hasMatchingConfig) {
                  const normalizedArea = normalizeStr(itemLocation);
                  if (!budgetMatchingAreas.find(a => normalizeStr(a) === normalizedArea)) {
                    budgetMatchingAreas.push(itemLocation);
                  }
                }
              }
            }
          }

          _suggestedAreas = budgetMatchingAreas
            .filter(a => normalizeStr(a) !== normalizeStr(resolvedLocation))
            .slice(0, 5);

          console.log(`[Integration] fallback: found ${budgetMatchingAreas.length} budget-matching areas, returning ${_suggestedAreas.length} suggestions`);
        } catch (err) {
          console.warn(`[Integration] fallback area scan failed: ${err.message}`);
          // Fall back to empty suggestions
          _suggestedAreas = [];
        }
      } else if (!budgetTooLow) {
        // For no_location_match (area doesn't exist), use catalog areas as fallback
        const _catalog = loadCatalog();
        const _cityData = _catalog.by_city?.[resolvedCity] || _catalog.by_city?.[city] || {};
        _suggestedAreas = (_cityData.micro_markets || _cityData.locations || [])
          .filter(a => normalizeStr(a) !== normalizeStr(resolvedLocation))
          .slice(0, 5);
      }

      pool.query(
        `INSERT INTO integration_requests (call_id,assistant_id,org_id,tool_name,params,bedrooms,result_count,error,duration_ms)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
        [call_id||null, assistant_id||null, org_id||null, 'search_properties',
         JSON.stringify({ city: resolvedCity, location: resolvedLocation, property_type: resolvedType, max_price, min_price, page: pageNum }),
         bedrooms || null, 0, budgetTooLow ? 'budget_too_low' : 'no_location_match', durationMs]
      ).catch(e => console.error(`[Integration] DB log FAILED (${e.code || 'ERR'}): ${e.message} â€” check integration_requests table exists and DATABASE_URL SSL config`));

      if (budgetTooLow) {
        return res.json({
          success:          true,
          data:             [],
          has_more:         false,
          current_page:     pageNum,
          budget_too_low:   true,            // area exists but user budget is too low
          area_min_price:   areaMinPrice,    // e.g. "45 L" â€” cheapest property in that area
          searched_location: location || resolvedLocation,
          suggested_areas:  _suggestedAreas, // areas WITHIN user's budget (not just nearby)
          search_scope:           'exact_locality',
          exact_location_searched: resolvedLocation || location || null,
          fallback_occurred:      false,
          message: `Properties in '${location || resolvedLocation}' start from ${areaMinPrice || 'a higher price point'}.`,
        });
      }

      return res.json({
        success: true,
        data: [],
        has_more: false,
        current_page: pageNum,
        no_location_match: true,
        searched_location: location || resolvedLocation,
        suggested_areas: _suggestedAreas,
        search_scope:           'exact_locality',
        exact_location_searched: resolvedLocation || location || null,
        fallback_occurred:      false,
        message: `No properties found in '${location || resolvedLocation}'.`,
      });
    }
  }

  // â”€â”€ Step 2: Deduplicate â€” remove already-shown slugs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  if (!resolvedLocation && filtered.length > 0) {
    if (project_category) {
      filtered = filtered.filter(matchesProjectCategory);
      console.log(`[Integration] category filter '${project_category}': ${filtered.length} matched`);
    }
    if (construction_state) {
      filtered = filtered.filter(matchesConstructionState);
      console.log(`[Integration] construction filter '${construction_state}': ${filtered.length} matched`);
    }
  }

  if (excludeSet.size > 0) {
    const before = filtered.length;
    filtered = filtered.filter(p => !excludeSet.has(p.Project_Slug));
    console.log(`[Integration] dedup: removed ${before - filtered.length} seen slugs, ${filtered.length} remain`);
  }

  // â”€â”€ Step 3: Normalize to voice-friendly listing shape, limit to PAGE_SIZE â”€â”€
  // IMPORTANT: search results intentionally include ONLY listing-level fields.
  // Detailed fields (amenities, sq-ft, possession date, RERA, description,
  // full config breakdowns) are deliberately excluded so the LLM must call
  // get_property_details to answer "tell me more about X" questions.
  // This also reduces token count and keeps LLM context lean.
  const normalize = p => {
    // Build a compact price + config summary (e.g. "2/3 BHK from 2.7 Cr")
    const configs = p.configurations || p.configs || [];
    const prices  = configs.map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0)).filter(n => n > 0);
    const minPriceVal = prices.length ? Math.min(...prices) : null;
    const starting_price = minPriceVal ? formatPriceINR(minPriceVal) : null;
    const types = [...new Set(configs.map(c => c.Config_Type).filter(Boolean))];
    const nums  = types.map(t => (t.match(/^\d+/) || [])[0]).filter(Boolean);
    const config_summary = (nums.length > 0 && types.every(t => /BHK/i.test(t)))
      ? `${nums.join('/')} BHK`
      : types.slice(0, 3).join('/') || null;

    return {
      // Identity â€” property_id MUST be present for get_property_details to work
      project_slug:    p.Project_Slug,
      name:            p.Project_Name_Original || p.Project_Name || p.Project_Name_New || null,
      project_name:    p.Project_Name_Original || p.Project_Name || p.Project_Name_New || null,
      // Location â€” enough to describe where the property is
      city:            p.City,
      location:        p.Location,
      micro_market:    p.Micro_Market,
      // Project basics â€” enough for a listing summary
      land_area:       p.Land_Area || p.LandArea || null,
      floors:          p.Number_Of_Floors || null,
      number_of_floors:p.Number_Of_Floors || null,
      description:     (p.Project_Desc || '').replace(/\\n/g, ' ').trim() || null,
      type:            p.Project_Type,
      construction:    p.State_Of_Construction || null,
      // sold_out:        p.Sold_Out || false,
      possession:      p.PosessionDate ? p.PosessionDate.slice(0, 10) : null,
      // rera:            p.RERA_No || p.RERA_Number || null,
      // address:         [p.Address_Line_1, p.Address_Line_2].filter(Boolean).join(', ') || null,
      // postal_code:     p.Postal_Code || null,
      // towers:          p.Number_Of_Towers || null,
      // units:           p.Number_Of_Units || null,
      category:        p.Project_Category || null,
      // state:           p.State || null,
      developer:       (p.developer || []).map(d => d.Connection_Name || d.name).filter(Boolean),
      developer_details: (p.developer || []).map(d => ({
        name: d.Connection_Name || d.name || null,
        id: d.Connection_ID || d.id || null,
      })),
      amenities:       (p.amenities || []).map(a => a.Attribute_Value || a.name).filter(Boolean),
      amenities_details: (p.amenities || []).map(a => ({
        name: a.Attribute_Value || a.name || null,
        id: a.Attribute_ID || a.id || null,
      })),
      // Price & config summary â€” agent uses these for the spoken property list
      starting_price,
      config_summary,
      configs:         (p.configurations || p.configs || []).map(c => ({
        type: c.Config_Type || null,
        super_builtup: c.Super_Built_Up_Area || null,
        carpet_area: c.Carpet_Area || null,
        builtup_area: c.Built_Up_Area || null,
        price: normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0) || null,
        base_price: c.Base_Price || null,
        active: c.Is_Active,
      })),
      // NOTE: amenities, sq-ft, possession date, RERA, description, developer,
      // full config breakdowns are NOT included here. The agent must call
      // get_property_details(property_id) to get those.
    };
  };

  // Valid (slug-bearing) filtered items â€” source for data, top_projects, and counts
  const validFiltered = filtered.filter(p => p.Project_Slug);

  // â”€â”€ Persona-driven sort â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Luxury buyers see most expensive first; Budget buyers see cheapest first.
  // Investor / default: leave JLL's natural order (recency/relevance).
  const minConfigPrice = item => {
    const prices = (item.configurations || item.configs || [])
      .map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0))
      .filter(n => n > 0);
    return prices.length ? Math.min(...prices) : null;
  };
  const minBudget = Number(min_price || 0) || null;
  const maxBudget = Number(max_price || 0) || null;
  const targetBudget = (minBudget != null && maxBudget != null)
    ? Math.round((minBudget + maxBudget) / 2)
    : (maxBudget ?? minBudget);
  const sortPref = (sort_preference || '').toLowerCase();
  if (targetBudget) {
    validFiltered.sort((a, b) => {
      const aPrice = minConfigPrice(a);
      const bPrice = minConfigPrice(b);
      if (aPrice == null && bPrice == null) return 0;
      if (aPrice == null) return 1;
      if (bPrice == null) return -1;
      const aDelta = Math.abs(aPrice - targetBudget);
      const bDelta = Math.abs(bPrice - targetBudget);
      if (aDelta !== bDelta) return aDelta - bDelta;
      return aPrice - bPrice;
    });
    console.log(`[Integration] Sorted by budget proximity (target=${targetBudget})`);
  } else if (sortPref === 'price_desc' || sortPref === 'price_asc') {
    validFiltered.sort((a, b) => {
      const aPrice = minConfigPrice(a);
      const bPrice = minConfigPrice(b);
      const aVal = aPrice == null ? (sortPref === 'price_desc' ? 0 : Infinity) : aPrice;
      const bVal = bPrice == null ? (sortPref === 'price_desc' ? 0 : Infinity) : bPrice;
      return sortPref === 'price_desc' ? bVal - aVal : aVal - bVal;
    });
    console.log(`[Integration] Sorted by ${sortPref} (persona-driven)`);
  }

  const data = validFiltered.slice(0, PAGE_SIZE).map(normalize);
  const bufferedPageItems = validFiltered.slice(PAGE_SIZE).map(normalize);

  // has_more=true when:
  //   (a) JLL's location filter was used and the raw page was full (20 results â†’ more pages likely)
  //   (b) Client-side accumulation found more than PAGE_SIZE matches (we accumulated extras)
  //   (c) Client-side accumulation hit PAGE_SIZE exactly and the last JLL page was still full
  //       (more properties in the city/area to scan â€” caller can increment page to get more)
  const jllPageFull  = raw.length >= 20;
  const hasMoreLocal = validFiltered.length > PAGE_SIZE;
  // When we used client-side accumulation we scanned up to MAX_ACCUMULATE_PAGES pages;
  // if we stopped because we found enough (not because pages ran out), there may be more.
  const has_more     = data.length > 0 && (jllPageFull || hasMoreLocal || usedClientSideFilter);
  const has_more_beyond_buffer = data.length > 0 && (jllPageFull || usedClientSideFilter);

  // â”€â”€ total_count â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Use JLL's own total if available (most accurate).
  // Otherwise fall back to validFiltered.length on this page.
  // NOTE: if has_more=true, the real total is higher than validFiltered.length â€”
  // the agent should say "at least N" when has_more is true.
  // When JLL pre-filtered by location (server-side), jllTotal is accurate.
  // When we fell back to client-side filtering, jllTotal is the city-wide total
  // (e.g. 3628 for all Chennai) â€” meaningless for the specific area.
  // Use validFiltered.length in that case.
  const total_count = (!usedClientSideFilter && jllTotal != null && Number.isFinite(Number(jllTotal)))
    ? Number(jllTotal)
    : validFiltered.length;

  // â”€â”€ top_projects â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // Top 3 from the current filtered set, with compact voice-readable fields.
  // Price is formatted as Indian currency: "1.2 Cr", "95 L", "45 L".
  // Config is a condensed string of unique types: "2/3 BHK", "Plots", etc.
  function formatPrice(val) {
    const n = Number(val);
    if (!n || !isFinite(n)) return null;
    if (n >= 10000000) {
      let crore = Math.floor(n / 10000000);
      let lakh = Math.round((n % 10000000) / 100000);
      if (lakh >= 100) {
        crore += 1;
        lakh = 0;
      }
      return lakh > 0 ? `${crore} Crore ${lakh} Lakh` : `${crore} Crore`;
    }
    if (n >= 100000)   return `${Math.round(n / 100000)} Lakh`;
    return `â‚¹${n.toLocaleString('en-IN')}`;
  }

  function summariseConfigs(configs) {
    if (!Array.isArray(configs) || configs.length === 0) return null;
    // Collect unique Config_Type values, drop nulls
    const types = [...new Set(
      configs.map(c => c.Config_Type).filter(Boolean)
    )];
    if (types.length === 0) return null;
    // Build "2/3 BHK" style string: extract numbers and append BHK if numeric
    const nums = types.map(t => (t.match(/^\d+/) || [])[0]).filter(Boolean);
    if (nums.length > 0 && types.every(t => /BHK/i.test(t))) {
      return `${nums.join('/')} BHK`;
    }
    return types.slice(0, 3).join('/');
  }

  function bestPrice(configs) {
    if (!Array.isArray(configs) || configs.length === 0) return null;
    const prices = configs
      .map(c => normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0))
      .filter(n => n > 0);
    if (prices.length === 0) return null;
    // Return the minimum (starting price)
    return formatPrice(Math.min(...prices));
  }

  const top_projects = validFiltered.slice(0, 3).map(p => ({
    name:   p.Project_Name_Original || p.Project_Name,
    price:  bestPrice(p.configurations || p.configs || []),
    config: summariseConfigs(p.configurations || p.configs || []),
  }));

  // min_price_available: the cheapest property price in this result set.
  // Used by the agent to inform callers when their budget is below available inventory,
  // e.g. "Properties here start from 45 L â€” that's above your 35 L budget."
  function cheapestPrice(items) {
    let min = Infinity;
    for (const p of items) {
      for (const c of (p.configurations || p.configs || [])) {
        const n = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
        if (n > 0 && n < min) min = n;
      }
    }
    return isFinite(min) ? formatPrice(min) : null;
  }
  const min_price_available = cheapestPrice(validFiltered);
  const resultCity = resolvedCity || city || null;
  const resultLocation = resolvedLocation || location || null;
  const resultCityData = loadCatalog().by_city?.[resultCity] || {};

  // â”€â”€ suggested_areas when data is empty â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // INTENTIONALLY empty []. Do NOT suggest raw catalog areas here.
  //
  // The old behaviour was to return the first 5 alphabetical micro-markets from
  // the catalog (Abiramapuram, Adambakkam, Adyar, â€¦) regardless of whether those
  // areas actually have the requested property type within budget. This caused the
  // agent to keep suggesting areas that also returned 0 results, creating an
  // infinite loop of useless suggestions.
  //
  // Correct behaviour: return [] so the agent system-prompt rule fires:
  //   "when search returns data=[] with no suggested_areas â†’ call areas_by_budget
  //    immediately to find real areas with this property type and budget."
  // The areas_by_budget endpoint actually scans JLL and only returns areas that
  // have matching inventory â€” it's the only source of truth for this.
  const zeroResultSuggestedAreas = [];
  const zeroResultsScope = data.length === 0 && resultLocation ? 'locality_only' : 'city_wide';

  const durationMs = Date.now() - t0;
  console.log(
    `[Integration] search_properties â† page=${pageNum} raw=${raw.length} ` +
    `filtered=${validFiltered.length} returned=${data.length} total=${total_count} ` +
    `bedrooms=${bedrooms || '-'} has_more=${has_more} error=${error || 'none'} (${durationMs}ms)`
  );
  if (data.length > 0) {
    _logBackendProperties('live search', data);
  }

  pool.query(
    `INSERT INTO integration_requests (call_id,assistant_id,org_id,tool_name,params,bedrooms,result_count,error,duration_ms)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`,
    [call_id||null, assistant_id||null, org_id||null, 'search_properties',
     JSON.stringify({ city, location, property_type, max_price, min_price, page: pageNum, exclude_count: excludeSet.size }),
     bedrooms || null, data.length, error, durationMs]
  ).catch(e => console.error(`[Integration] DB log FAILED (${e.code || 'ERR'}): ${e.message} â€” check integration_requests table exists and DATABASE_URL SSL config`));

  // â”€â”€ Explicit search metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // These fields are consumed by the voice agent's response-validation guard to
  // verify that results actually match the user's requested location/scope before
  // speaking them. Without this metadata the guard has to guess from result fields.
  //
  // search_scope:
  //   "exact_locality"  â€” we sent location= to JLL and got direct matches
  //   "client_filtered" â€” JLL returned 0 with location=, we fetched city-wide and
  //                       applied client-side filter (tokenised match); results may
  //                       be a superset â€” agent should not over-promise accuracy
  //   "city_wide"       â€” no location filter was used at all (city-wide query)
  //
  // fallback_occurred:  true when we silently dropped the location param to avoid
  //                     returning 0 results (non-catalog locality). Lets the agent
  //                     caveat its response: "I searched across Chennai for you."
  const _searchScope = !resultLocation
    ? 'city_wide'
    : usedClientSideFilter
      ? 'client_filtered'
      : 'exact_locality';

  const _responsePayload = {
    success:            !error,
    data,
    buffered_page_items: bufferedPageItems,
    has_more,
    has_more_beyond_buffer,
    current_page:       pageNum,
    total_count,
    top_projects,
    searched_city:      resultCity,
    searched_location:  resultLocation,
    searched_property_type: resolvedType || property_type || null,
    zero_results_scope: zeroResultsScope,
    suggested_areas:    zeroResultSuggestedAreas,
    min_price_available,  // cheapest property price in results - for budget-mismatch messaging
    // Explicit search-scope metadata for response-validation guard in voice agent
    search_scope:           _searchScope,
    exact_location_searched: resultLocation || null,
    fallback_occurred:      usedClientSideFilter && !!resultLocation,
  };

  // Cache successful, non-empty responses (skip caching errors or pagination dedup)
  if (!error && !_skipCache && data.length > 0) {
    _setCache(_ck, _responsePayload);
  }

  res.json(_responsePayload);
}

router.get('/proxy/search', handleSearch);
router.post('/proxy/search', handleSearch);

// â”€â”€â”€ GET /api/integration/proxy/property-resolve â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
router.get('/proxy/property-resolve', async (req, res) => {
  const t0 = Date.now();
  const {
    name = '',
    city = 'Chennai',
    location_hint = '',
    variants = '',
    limit = '3',
    call_id,
    assistant_id,
    org_id,
  } = req.query;

  const cleanedName = String(name || '').trim();
  const parsedVariants = _parsePropertyResolveVariants(variants);
  const resolveQueries = _propertyResolveQueries(cleanedName, parsedVariants);
  const resolvedCity = resolveCatalogName(city, 'city') || city || 'Chennai';
  const resultLimit = Math.max(1, Math.min(3, Number(limit) || 3));
  const cacheKey = _propertyResolveCacheKey({
    city: resolvedCity,
    name: cleanedName,
    location_hint,
    variants: parsedVariants,
  });
  const cached = _getTimedCache(_propertyResolveCache, cacheKey, PROPERTY_RESOLVE_TTL_MS);
  if (cached) {
    console.log(`[Integration] property_resolve cache HIT city=${resolvedCity} name="${cleanedName}"`);
    return res.json(cached);
  }

  if (!cleanedName) {
    return res.json({ success: false, match_type: 'none', candidates: [], error: 'Property name is required' });
  }

  console.log(
    `[Integration] property_resolve â†’ city=${resolvedCity} name="${cleanedName}" ` +
    `location_hint=${location_hint || '-'} variants=${parsedVariants.length} call=${call_id || '-'}`
  );

  let error = null;
  let candidates = [];
  try {
    const index = await _getPropertyIndex(resolvedCity);
    candidates = index
      .map(item => {
        const scored = _scorePropertyCandidateForQueries(item, resolveQueries, String(location_hint || ''));
        return { ...item, _score: scored.score, _matched_query: scored.matchedQuery };
      })
      .filter(item => item._score >= 40)
      .sort((a, b) => {
        if (b._score !== a._score) return b._score - a._score;
        const aHint = normalizeStr([a.location || '', a.micro_market || ''].join(' '));
        const bHint = normalizeStr([b.location || '', b.micro_market || ''].join(' '));
        const hint = normalizeStr(location_hint || '');
        if (hint) {
          const aBoost = aHint.includes(hint) ? 1 : 0;
          const bBoost = bHint.includes(hint) ? 1 : 0;
          if (bBoost !== aBoost) return bBoost - aBoost;
        }
        return String(a.name || '').localeCompare(String(b.name || ''));
      })
      .slice(0, resultLimit + 2);

    // Fallback: if no match from the cached city-wide index and a location hint is present,
    // do a smaller fresh location-filtered build to catch properties outside the cached slice.
    if (!candidates.length && location_hint) {
      const narrowed = await _buildPropertyIndex(resolvedCity, String(location_hint));
      candidates = narrowed
        .map(item => {
          const scored = _scorePropertyCandidateForQueries(item, resolveQueries, String(location_hint || ''));
          return { ...item, _score: scored.score, _matched_query: scored.matchedQuery };
        })
        .filter(item => item._score >= 35)
        .sort((a, b) => b._score - a._score)
        .slice(0, resultLimit + 2);
    }
  } catch (err) {
    error = err.message;
    console.error(`[Integration] property_resolve error: ${error}`);
  }

  const best = candidates[0] || null;
  const second = candidates[1] || null;
  let match_type = 'none';
  if (best) {
    if (!second) {
      match_type = 'single';
    } else {
      const scoreGap = best._score - second._score;
      match_type = (best._score >= 96 && scoreGap >= 8) ? 'single' : 'multiple';
    }
  }

  const response = {
    success: !error && !!best,
    match_type,
    searched_name: cleanedName,
    searched_variants: parsedVariants,
    searched_city: resolvedCity,
    location_hint: location_hint || null,
    candidates: candidates.slice(0, resultLimit).map(item => ({
      property_id: item.property_id,
      project_slug: item.project_slug,
      name: item.name,
      location: item.location,
      micro_market: item.micro_market,
      city: item.city,
      bhk: item.bhk,
      price: item.price,
      score: item._score,
      matched_query: item._matched_query || cleanedName,
    })),
    error: error || null,
  };

  const durationMs = Date.now() - t0;
  console.log(
    `[Integration] property_resolve â† match_type=${match_type} ` +
    `count=${response.candidates.length} error=${error || 'none'} (${durationMs}ms)`
  );
  if (response.candidates.length) {
    console.log(
      `[BackendPropertyResolve] ${response.candidates.map(_candidateDisplayBits).join(' | ')}`
    );
  }

  pool.query(
    `INSERT INTO integration_requests
       (call_id, assistant_id, org_id, tool_name, params, result_count, error, duration_ms)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
    [
      call_id || null,
      assistant_id || null,
      org_id || null,
      'resolve_property_by_name',
      JSON.stringify({
        name: cleanedName,
        city: resolvedCity,
        location_hint: location_hint || null,
        variants: parsedVariants,
      }),
      response.candidates.length,
      error,
      durationMs,
    ]
  ).catch(e => console.error(`[Integration] DB log FAILED (${e.code || 'ERR'}): ${e.message}`));

  _setTimedCache(_propertyResolveCache, cacheKey, response, 200);
  res.json(response);
});

// â”€â”€â”€ GET /api/integration/proxy/property/:slug â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
router.get('/proxy/property/:slug', async (req, res) => {
  const t0 = Date.now();
  const { slug } = req.params;
  const { call_id, assistant_id, org_id } = req.query;

  console.log(`[Integration] get_property_details â†’ slug=${slug} call=${call_id || '-'}`);

  let data = null, error = null;
  try {
    const jllRes = await fetch(
      `${JLL_BASE}/api/user/project/${encodeURIComponent(slug)}`,
      { timeout: 10000 }
    );
    const json = await jllRes.json();
    const p = json?.data || json;
    if (p && p.Project_Name) {
      const rawDesc = (p.Project_Desc || '').replace(/\\n/g, ' ').trim();
      data = {
        // Identity
        property_id:        p.Project_Slug,
        name:               p.Project_Name_Original || p.Project_Name,
        full_name:          p.Project_Name,
        name_with_config:   p.Project_Name_With_Config,
        seo_name:           p.Project_Name_New,
        // Location
        city:               p.City,
        state:              p.State,
        location:           p.Location,
        micro_market:       p.Micro_Market,
        address:            [p.Address_Line_1, p.Address_Line_2].filter(Boolean).join(', ') || null,
        postal_code:        p.Postal_Code,
        // Project details
        type:               p.Project_Type,
        category:           p.Project_Category,
        construction:       p.State_Of_Construction,
        floors:             p.Number_Of_Floors,
        towers:             p.Number_Of_Towers,
        units:              p.Number_Of_Units,
        possession:         p.PosessionDate ? p.PosessionDate.slice(0, 10) : null,
        rera:               _extractProjectRera(p),
        sold_out:           p.Sold_Out || false,
        description:        rawDesc || null,
        // Developer
        developer:          (p.developer || []).map(d => d.Connection_Name).filter(Boolean),
        // Configurations â€” full details (price=null when no price data available)
        configs: (p.configurations || p.configs || []).map(c => {
          const rawPrice = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
          return {
            type:          c.Config_Type,
            super_builtup: c.Super_Built_Up_Area,
            carpet_area:   c.Carpet_Area,
            builtup_area:  c.Built_Up_Area,
            price:         rawPrice > 0 ? rawPrice : null,
            base_price:    c.Base_Price || null,
            active:        c.Is_Active,
          };
        }),
        // Amenities
        amenities: (p.amenities || []).map(a => a.Attribute_Value).filter(Boolean),
        // NOTE: Media files (images, brochures) are intentionally excluded.
        // This is a voice agent â€” URLs have no meaning in audio and the LLM
        // will read them aloud if included. Never add image/file URLs here.
      };
      // Compute starting_price: cheapest config with a valid price > 0.
      // Formatted as human-readable Indian currency ("45 L", "1.3 Cr").
      // null when no config has price data â€” agent must NOT say "starts from 0".
      const allConfigPrices = (data.configs || [])
        .map(c => c.price)
        .filter(n => n != null && n > 0);
      data.starting_price = allConfigPrices.length > 0
        ? formatPriceINR(Math.min(...allConfigPrices))
        : null;
    }
  } catch (err) {
    error = err.message;
  }

  const durationMs = Date.now() - t0;
  console.log(`[Integration] get_property_details â† found=${!!data} error=${error || 'none'} (${durationMs}ms)`);

  pool.query(
    `INSERT INTO integration_requests
       (call_id, assistant_id, org_id, tool_name, params, result_count, error, duration_ms)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
    [call_id || null, assistant_id || null, org_id || null, 'get_property_details',
     JSON.stringify({ slug }), data ? 1 : 0, error, durationMs]
  ).catch(e => console.error(`[Integration] DB log FAILED (${e.code || 'ERR'}): ${e.message} â€” check integration_requests table exists and DATABASE_URL SSL config`));

  if (data) {
    console.log(
      `[BackendPropertyDetails] ${data.name || '-'} (${data.type || '-'}) @ ` +
      `${data.location || data.micro_market || data.city || '-'} @ ${data.starting_price || '-'} ` +
      `| possession=${data.possession || '-'} | amenities=${Array.isArray(data.amenities) ? data.amenities.length : 0} ` +
      `| configs=${Array.isArray(data.configs) ? data.configs.length : 0}`
    );
  }
  if (!data) return res.json({ success: false, error: error || 'Property not found' });
  res.json({ success: true, data });
});

// â”€â”€â”€ Proximity helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Static lat/lon for known micro-markets per city.
// Source: Nominatim geocoding (run once offline). Add new entries as catalog grows.
// For areas NOT in this map, distance filtering is skipped (they are always included).
const AREA_COORDS = {
  Chennai: {
    'Anna Nagar':       [13.0850, 80.2101],
    'Adyar':            [13.0012, 80.2565],
    'T Nagar':          [13.0358, 80.2330],
    'T. Nagar':         [13.0358, 80.2330],
    'Nungambakkam':     [13.0569, 80.2425],
    'Kodambakkam':      [13.0495, 80.2246],
    'Guindy':           [13.0067, 80.2206],
    'Velachery':        [12.9654, 80.2207],
    'Pallikaranai':     [12.9382, 80.2030],
    'Medavakkam':       [12.9246, 80.1967],
    'Madipakkam':       [12.9511, 80.2024],
    'Perungudi':        [12.9634, 80.2437],
    'Thoraipakkam':     [12.9387, 80.2357],
    'Sholinganallur':   [12.9010, 80.2275],
    'Perumbakkam':      [12.9030, 80.1940],
    'Navalur':          [12.8480, 80.2271],
    'Siruseri':         [12.8167, 80.2270],
    'Thazhambur':       [12.8451, 80.2089],
    'Kelambakkam':      [12.7884, 80.2184],
    'Thaiyur':          [12.8206, 80.2232],
    'Vellavedu':        [12.8050, 80.2070],
    'OMR':              [12.9000, 80.2273],
    'ECR':              [12.8333, 80.2500],
    'GST Road':         [12.9000, 80.1000],
    'Tambaram':         [12.9249, 80.1000],
    'Chrompet':         [12.9516, 80.1430],
    'Pallavaram':       [12.9676, 80.1491],
    'Urapakkam':        [12.8662, 80.0822],
    'Porur':            [13.0368, 80.1570],
    'Valasaravakkam':   [13.0467, 80.1758],
    'Saligramam':       [13.0490, 80.1989],
    'Poonamallee':      [13.0468, 80.1086],
    'Ambattur':         [13.1142, 80.1548],
    'Perambur':         [13.1133, 80.2420],
    'Kolathur':         [13.1200, 80.2250],
    'West Chennai':     [13.0827, 80.1716],
    'North Chennai':    [13.1500, 80.2800],
    'Nanganallur':      [12.9737, 80.1855],
    'Keelkattalai':     [12.9529, 80.1875],
    'Mudichur':         [12.9126, 80.0628],
  },
  Bengaluru: {
    'Whitefield':       [12.9698, 77.7500],
    'Electronic City':  [12.8399, 77.6770],
    'Sarjapur':         [12.8599, 77.7862],
    'Koramangala':      [12.9279, 77.6271],
    'HSR Layout':       [12.9116, 77.6474],
    'BTM Layout':       [12.9166, 77.6101],
    'Marathahalli':     [12.9591, 77.6974],
    'Bellandur':        [12.9256, 77.6760],
    'Hebbal':           [13.0359, 77.5970],
    'Yelahanka':        [13.1007, 77.5963],
  },
  Hyderabad: {
    'Gachibowli':       [17.4401, 78.3489],
    'Kondapur':         [17.4700, 78.3596],
    'Madhapur':         [17.4483, 78.3915],
    'HITEC City':       [17.4435, 78.3772],
    'Manikonda':        [17.4052, 78.3899],
    'Kukatpally':       [17.4948, 78.3996],
    'Bachupally':       [17.5406, 78.3726],
    'Nallagandla':      [17.4533, 78.3299],
  },
  Mumbai: {
    'Thane':            [19.2183, 72.9781],
    'Navi Mumbai':      [19.0368, 73.0158],
    'Andheri':          [19.1136, 72.8697],
    'Powai':            [19.1176, 72.9060],
    'Goregaon':         [19.1663, 72.8526],
    'Kandivali':        [19.2043, 72.8490],
  },
};

/** Haversine straight-line distance in km between two lat/lon points. */
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2
    + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180)
    * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Filter areas by proximity to `nearLocation`, expanding radius from
 * `startKm` in `stepKm` increments until at least `minResults` areas are found.
 * Areas whose coordinates are unknown are kept in results (no distance info).
 * Returns { filtered: area[], radiusUsed: number }
 */
/** Normalize an area name for AREA_COORDS lookup: strip dots, collapse spaces. */
function _normalizeAreaKey(name) {
  return (name || '').replace(/\./g, ' ').replace(/\s+/g, ' ').trim();
}

function filterByProximity(areas, city, nearLocation, startKm = 5, stepKm = 5, maxKm = 30, minResults = 3) {
  const cityCoords = AREA_COORDS[city] || {};
  // Try exact key first, then normalized (handles "T.Nagar" â†’ "T Nagar", "T. Nagar" â†’ "T Nagar")
  const _normNear = _normalizeAreaKey(nearLocation);
  const originCoords = cityCoords[nearLocation]
    || cityCoords[_normNear]
    || Object.entries(cityCoords).find(([k]) => _normalizeAreaKey(k) === _normNear)?.[1];
  if (!originCoords) {
    // Origin area not in static map â€” return all areas (can't compute distance)
    console.warn(`[Integration] proximity: '${nearLocation}' not in AREA_COORDS for ${city} â€” skipping distance filter`);
    return { filtered: areas, radiusUsed: null };
  }
  const [lat0, lon0] = originCoords;

  for (let radius = startKm; radius <= maxKm; radius += stepKm) {
    const filtered = areas.filter(a => {
      const _normA = _normalizeAreaKey(a.name);
      const aCoords = cityCoords[a.name]
        || cityCoords[_normA]
        || Object.entries(cityCoords).find(([k]) => _normalizeAreaKey(k) === _normA)?.[1];
      if (!aCoords) return true; // unknown area â€” include by default
      const dist = haversineKm(lat0, lon0, aCoords[0], aCoords[1]);
      a._distance_km = Math.round(dist * 10) / 10; // attach for sorting
      return dist <= radius;
    });
    if (filtered.length >= minResults || radius >= maxKm) {
      // Sort known-distance areas first, closest first
      filtered.sort((a, b) => {
        if (a._distance_km == null && b._distance_km == null) return b.property_count - a.property_count;
        if (a._distance_km == null) return 1;
        if (b._distance_km == null) return -1;
        return a._distance_km - b._distance_km;
      });
      console.log(`[Integration] proximity filter: ${filtered.length} areas within ${radius}km of '${nearLocation}'`);
      return { filtered, radiusUsed: radius };
    }
  }
  return { filtered: areas, radiusUsed: null };
}

// â”€â”€â”€ GET /api/integration/proxy/areas-by-budget â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Returns areas/locations that have properties WITHIN the specified budget range.
// Optionally filters by proximity to near_location, expanding radius until results found.
async function handleAreasByBudget(req, res) {
  const t0 = Date.now();
  const p = Object.assign({}, req.query, req.body);
  const {
    city, property_type, max_price, min_price,
    near_location, radius_km,
    page = '1',
    limit = '3',
    call_id, assistant_id, org_id,
  } = p;

  const pageNum = Math.max(1, parseInt(page) || 1);
  const limitNum = Math.min(Math.max(1, parseInt(limit) || 3), 10); // Cap at 10

  if (!city) {
    return res.status(400).json({ success: false, error: 'city is required' });
  }

  const resolvedCity = resolveCatalogName(city, 'city') || city || '';
  const resolvedType = property_type ? (resolveCatalogName(property_type, 'property_type', resolvedCity) || property_type) : '';
  
  const maxPrice = max_price ? Number(max_price) : null;
  const minPrice = min_price ? Number(min_price) : null;

  console.log(
    `[Integration] areas_by_budget â†’ city=${resolvedCity} ` +
    `type=${resolvedType || '-'} budget=${minPrice || 0}-${maxPrice || 'âˆž'} ` +
    `call=${call_id || '-'}`
  );

  // â”€â”€ Cache check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const _ack = _areasCacheKey(resolvedCity, resolvedType, minPrice, maxPrice);
  let budgetMatchingAreas = _getAreasCache(_ack);
  const cacheHit = !!budgetMatchingAreas;

  if (!budgetMatchingAreas) {
    // Scan JLL pages in parallel batches of 5 instead of sequentially.
    // Sequential was: 15 Ã— ~70ms = ~1050ms worst-case.
    // Parallel batches: ceil(15/5) Ã— ~70ms = ~210ms worst-case, plus early exit.
    budgetMatchingAreas = [];
    const MAX_SCAN_PAGES = 15;
    const BATCH_SIZE = 5;
    const EARLY_EXIT_AREAS = 20; // enough unique areas â€” stop scanning

    const _buildQs = (pg) => {
      const qs = new URLSearchParams({ city: resolvedCity || '', page: String(pg), limit: '20' });
      if (resolvedType) qs.set('property_type', resolvedType);
      if (maxPrice)     qs.set('max_price', String(maxPrice));
      if (minPrice)     qs.set('min_price', String(minPrice));
      return qs;
    };

    const _accumulateItems = (items) => {
      for (const item of items) {
        const itemLocation = item.Location || item.Micro_Market || '';
        if (!itemLocation) continue;
        const configs = item.configurations || item.configs || [];
        const hasMatchingConfig = configs.some(c => {
          const price = normalizeProjectPriceINR(c.FinalPrice || c.All_Price || 0);
          if (price <= 0) return false;
          if (minPrice && price < minPrice) return false;
          if (maxPrice && price > maxPrice) return false;
          return true;
        });
        if (hasMatchingConfig) {
          const normalizedArea = normalizeStr(itemLocation);
          const existing = budgetMatchingAreas.find(a => normalizeStr(a.name) === normalizedArea);
          if (existing) existing.property_count += 1;
          else budgetMatchingAreas.push({ name: itemLocation, property_count: 1 });
        }
      }
    };

    try {
      let done = false;
      for (let batchStart = 1; batchStart <= MAX_SCAN_PAGES && !done; batchStart += BATCH_SIZE) {
        const pages = [];
        for (let pg = batchStart; pg < batchStart + BATCH_SIZE && pg <= MAX_SCAN_PAGES; pg++) {
          pages.push(pg);
        }

        // Fetch all pages in this batch concurrently
        const results = await Promise.all(
          pages.map(pg =>
            fetch(`${JLL_BASE}/api/user/search/projects?${_buildQs(pg)}`, { timeout: 8000 })
              .then(r => r.json())
              .then(j => Array.isArray(j?.data) ? j.data : [])
              .catch(() => [])
          )
        );

        for (let i = 0; i < results.length; i++) {
          const items = results[i];
          if (items.length === 0) { done = true; break; }
          _accumulateItems(items);
          if (items.length < 20) { done = true; break; } // last page in JLL
        }

        // Early exit if we already have plenty of unique areas
        if (budgetMatchingAreas.length >= EARLY_EXIT_AREAS) done = true;
      }
    } catch (err) {
      console.error(`[Integration] areas_by_budget scan failed: ${err.message}`);
    }

    _setAreasCache(_ack, budgetMatchingAreas);
  }

  // Sort by property count descending (base sort before proximity filter)
  budgetMatchingAreas.sort((a, b) => b.property_count - a.property_count);

  // â”€â”€ Proximity filter (near_location) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  // If caller provided near_location, filter to areas within expanding radius
  // starting at radius_km (default 5km), expanding by 5km steps until â‰¥3 found.
  let finalAreas = budgetMatchingAreas;
  let radiusUsed = null;
  const resolvedNearLocation = near_location
    ? (resolveCatalogName(near_location, 'micro_market', resolvedCity) || near_location)
    : null;

  if (resolvedNearLocation) {
    const startRadius = radius_km ? Number(radius_km) : 5;
    const { filtered, radiusUsed: r } = filterByProximity(
      budgetMatchingAreas, resolvedCity, resolvedNearLocation,
      startRadius, 5, 30, 3
    );
    finalAreas = filtered;
    radiusUsed = r;
    console.log(
      `[Integration] areas_by_budget proximity: near='${resolvedNearLocation}' ` +
      `radius=${radiusUsed}km â†’ ${finalAreas.length} areas`
    );
  }

  const durationMs = Date.now() - t0;
  console.log(
    `[Integration] areas_by_budget â† ${cacheHit ? 'CACHE HIT' : 'scanned'} ${finalAreas.length} areas ` +
    `page=${pageNum} limit=${limitNum} (${durationMs}ms)`
  );

  // Paginate results
  const startIdx = (pageNum - 1) * limitNum;
  const paginatedAreas = finalAreas.slice(startIdx, startIdx + limitNum);
  const hasMore = finalAreas.length > startIdx + limitNum;

  pool.query(
    `INSERT INTO integration_requests (call_id,assistant_id,org_id,tool_name,params,result_count,error,duration_ms)
     VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
    [call_id||null, assistant_id||null, org_id||null, 'areas_by_budget',
     JSON.stringify({ city: resolvedCity, property_type: resolvedType, max_price: maxPrice, min_price: minPrice, near_location: resolvedNearLocation, page: pageNum, limit: limitNum }),
     finalAreas.length, null, durationMs]
  ).catch(e => console.error(`[Integration] DB log FAILED: ${e.message}`));

  res.json({
    success: true,
    areas: paginatedAreas,
    total_areas: finalAreas.length,
    current_page: pageNum,
    limit: limitNum,
    has_more: hasMore,
    searched_city: resolvedCity,
    near_location: resolvedNearLocation || null,
    radius_km: radiusUsed,
    budget_range: { min: minPrice, max: maxPrice },
  });
}

router.get('/proxy/areas-by-budget', handleAreasByBudget);
router.post('/proxy/areas-by-budget', handleAreasByBudget);

// â”€â”€â”€ Catalog management endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// GET /api/integration/catalog
// Returns the loaded catalog (locations, micro_markets, property_types per city).
// Useful for the frontend to show available filter values or for debugging.
router.get('/catalog', (req, res) => {
  const catalog = loadCatalog();
  res.json({
    success: true,
    synced_at: catalog.synced_at,
    cities: catalog.cities,
    property_types: catalog.property_types,
    by_city: catalog.by_city,
  });
});

// POST /api/integration/catalog/reload
// Hot-reloads the catalog from disk without restarting the server.
// Called automatically by sync-jll-catalog.js after it finishes writing the file.
router.post('/catalog/reload', (req, res) => {
  const catalog = reloadCatalog();
  console.log(`[Catalog] Hot-reloaded: ${catalog.cities.length} cities`);
  res.json({
    success: true,
    message: `Catalog reloaded: ${catalog.cities.length} cities, synced_at=${catalog.synced_at}`,
  });
});

module.exports = router;
