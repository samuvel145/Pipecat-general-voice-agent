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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ JLL Integration Proxy listening on http://localhost:${PORT}`);
  console.log(`   Routes mounted at: /api/integration/proxy/*`);
});
