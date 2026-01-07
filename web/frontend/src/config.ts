// dynamic configuration for API URL
// In production (Docker/Cloud Run), NEXT_PUBLIC_API_URL is injected at build time.
// In local dev, it falls back to localhost:8001.

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export default API_BASE_URL;
