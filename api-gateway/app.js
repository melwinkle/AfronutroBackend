const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();
const PORT = process.env.PORT || 3000;

// Proxy middleware options
const options = {
  target: 'http://localhost:8000', // Changed from 'http://django:8000'
  changeOrigin: true,
  pathRewrite: {
    '^/api': '', // remove /api prefix to match Django's root URL
  },
  onProxyReq: (proxyReq, req, res) => {
    // Forward original headers
    proxyReq.setHeader('X-Forwarded-Host', req.get('host'));
    proxyReq.setHeader('X-Forwarded-For', req.headers['x-forwarded-for'] || req.connection.remoteAddress);
    proxyReq.setHeader('X-Forwarded-Proto', req.protocol);
  },
  onError: (err, req, res) => {
    console.error('Proxy Error:', err);
    res.status(500).json({ error: 'Proxy error', details: err.message });
  },
  logLevel: 'debug' // Add this line to enable detailed logging
};

// Logging middleware
app.use((req, res, next) => {
  console.log(`Received request: ${req.method} ${req.url}`);
  next();
});

// Create the proxy middleware
const apiProxy = createProxyMiddleware(options);

// Use the proxy middleware for '/api' routes
app.use('/api', apiProxy);

// Add a health check route
app.get('/health', (req, res) => {
  res.status(200).json({ status: 'OK' });
});

app.listen(PORT, () => {
  console.log(`API Gateway running on port ${PORT}`);
  console.log(`Proxying requests to: ${options.target}`);
});