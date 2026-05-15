/**
 * proxy-server.js
 * Standalone Express server that mounts the JLL integration router.
 *
 * Usage:  node proxy-server.js
 * Listens on http://localhost:3000
 *
 * The Python voice agent expects:
 *   JLL_PROXY_URL = http://localhost:3000/api/integration
 */

require('dotenv').config();

const express = require('express');
const app     = express();

// Parse JSON bodies
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Mount the JLL integration router
const integrationRouter = require('./integration (1).js');
app.use('/api/integration', integrationRouter);

// Health check
app.get('/', (_req, res) => res.json({ status: 'JLL proxy running', port: PORT }));

// Catalog hot-reload (called by sync-catalog.js after writing new catalog)
app.post('/api/integration/catalog/reload', (_req, res) => {
  try {
    const integration = require('./integration (1).js');
    if (typeof integration.reloadCatalog === 'function') integration.reloadCatalog();
    console.log('[Proxy] Catalog hot-reloaded');
    res.json({ success: true });
  } catch (e) {
    res.json({ success: false, error: e.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ JLL Integration Proxy listening on http://localhost:${PORT}`);
  console.log(`   Routes mounted at: /api/integration/proxy/*`);
});
