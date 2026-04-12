const isProduction = process.env.NODE_ENV === "production";

// NEXT_PUBLIC_API_URL takes priority.
// Fallback: current EC2 backend in production, localhost in development.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (isProduction
    ? "https://34.229.139.86.sslip.io"
    : "http://localhost:8000");

export const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
export default API_BASE_URL;
