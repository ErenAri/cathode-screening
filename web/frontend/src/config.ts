const isProduction = process.env.NODE_ENV === "production";

// NEXT_PUBLIC_API_URL takes priority (set in Vercel/Render dashboard).
// Fallback: Render backend in production, localhost in development.
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  (isProduction
    ? "https://cathode-screening-api.onrender.com"
    : "http://localhost:8000");

export const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
export default API_BASE_URL;
